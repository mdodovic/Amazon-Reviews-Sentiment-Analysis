import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dataloader import load_and_preprocess_data_no_subtract

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path_to_model = './models/new_bert_amazon_food_review'
path_to_file = "./dataset/" + "test.txt"

texts, labels = load_and_preprocess_data_no_subtract(path_to_file)

# Define hyperparameters
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # Adjusted learning rate for BERT models
EPOCHS = 15

# Load a pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(labels))).to(device)

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

X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.02, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_ids, X_train_masks = tokenize_and_pad(X_train, tokenizer, MAX_SEQ_LENGTH)
X_val_ids, X_val_masks = tokenize_and_pad(X_val, tokenizer, MAX_SEQ_LENGTH)
X_test_ids, X_test_masks = tokenize_and_pad(X_test, tokenizer, MAX_SEQ_LENGTH)

# Create DataLoader for training and validation
train_data = TensorDataset(X_train_ids.to(device), X_train_masks.to(device), torch.tensor(y_train, dtype=torch.long).to(device))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = TensorDataset(X_val_ids.to(device), X_val_masks.to(device), torch.tensor(y_val, dtype=torch.long).to(device))
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Initialize the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss().to(device)

# Initialize lists to store training and validation losses and accuracies
train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []

# Early stopping variables
best_val_loss = float('inf')  # Initialize with a large value
patience = 5  # Number of epochs to wait for improvement
wait = 0  # Counter for consecutive epochs without improvement

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.logits, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calculate and print average loss and accuracy for this epoch
    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    # Evaluation on the validation set
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            total_val += labels.size(0)
            correct_val += (predictions == labels).sum().item()

    # Calculate accuracy on the validation set
    accuracy = 100 * correct_val / total_val
    validation_accuracies.append(accuracy)

    # Calculate and store validation loss
    model.train()
    total_val_loss = 0
    for val_batch in DataLoader(TensorDataset(X_test_ids.to(device), X_test_masks.to(device), torch.tensor(y_test, dtype=torch.long).to(device)), batch_size=BATCH_SIZE):
        val_input_ids, val_attention_mask, val_labels = val_batch
        val_outputs = model(val_input_ids, val_attention_mask)
        val_loss = criterion(val_outputs.logits, val_labels)
        total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(DataLoader(TensorDataset(X_test_ids, X_test_masks, torch.tensor(y_test, dtype=torch.long).to(device)), batch_size=BATCH_SIZE))
    validation_losses.append(avg_val_loss)

    print(f"Validation loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0  # Reset the wait counter since there's improvement
        # save model
        model.save_pretrained(path_to_model)

    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping after {epoch + 1} epochs without improvement.")
            break  # Stop training

# Load the best model from the saved checkpoint
model = BertForSequenceClassification.from_pretrained(path_to_model)

# Evaluate the model on the test set
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for batch in DataLoader(TensorDataset(X_test_ids, X_test_masks, torch.tensor(y_test, dtype=torch.long).to(device)), batch_size=BATCH_SIZE):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        total_test += labels.size(0)
        correct_test += (predictions == labels).sum().item()

# Calculate accuracy on the test set
test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot the training and validation loss and accuracy
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig("BERT.png", dpi=90)
plt.show()
