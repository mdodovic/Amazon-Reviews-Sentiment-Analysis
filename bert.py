import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import load_and_preprocess_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path_to_file = "./dataset/" + "test.txt"

texts, labels = load_and_preprocess_data(path_to_file)

# Define hyperparameters
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-8
EPOCHS = 100

# Load a pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Define the sentiment analysis model
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = pretrained_model
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = torch.mean(last_hidden_state, dim=1)  # Mean pooling
        logits = self.fc(pooler_output)
        return logits

# Create train and test datasets
# Replace this with your dataset loading and preprocessing code

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize and pad sequences
def tokenize_and_pad(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True,
        ).to(device) 
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

X_train_ids, X_train_masks = tokenize_and_pad(X_train, tokenizer, MAX_SEQ_LENGTH)
X_test_ids, X_test_masks = tokenize_and_pad(X_test, tokenizer, MAX_SEQ_LENGTH)

# Create DataLoader for batching
train_data = TensorDataset(X_train_ids.to(device), X_train_masks.to(device), torch.tensor(y_train, dtype=torch.long).to(device))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model
num_classes = len(np.unique(labels))
print(num_classes)
model = SentimentClassifier(model, num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss().to(device)

# Initialize lists to store training and validation losses
train_losses = []
validation_losses = []
validation_accuracies = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Calculate and print average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    train_losses.append(avg_loss)

    # Evaluation on the validation set
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(X_test_ids, X_test_masks), batch_size=BATCH_SIZE):
            input_ids, attention_mask = batch
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            y_pred.extend(predictions.cpu().numpy())

    # Calculate accuracy on the validation set
    accuracy = accuracy_score(y_test, y_pred)
    validation_accuracies.append(accuracy)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Calculate and store validation loss
    model.train()
    total_val_loss = 0
    for val_batch in DataLoader(TensorDataset(X_test_ids.to(device), X_test_masks.to(device), torch.tensor(y_test, dtype=torch.long).to(device)), batch_size=BATCH_SIZE):
        val_input_ids, val_attention_mask, val_labels = val_batch
        val_logits = model(val_input_ids, val_attention_mask)
        val_loss = criterion(val_logits, val_labels)
        total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(DataLoader(TensorDataset(X_test_ids, X_test_masks, torch.tensor(y_test, dtype=torch.long).to(device)), batch_size=BATCH_SIZE))
    validation_losses.append(avg_val_loss)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, EPOCHS + 1), validation_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.show()
