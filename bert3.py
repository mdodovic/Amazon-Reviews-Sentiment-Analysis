import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataloader import load_and_preprocess_data

# Set up GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path_to_model = './models/bert_amazon_food_review'
path_to_file = "./dataset/" + "test.txt"

# Load and preprocess data
texts, labels = load_and_preprocess_data(path_to_file)

# Hyperparameters
batch_size = 8
learning_rate = 5e-5
num_epochs = 3
patience = 5
num_labels = 5  # Assuming 5 labels (0.0, 1.0, 2.0, 3.0, 4.0)

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Tokenize and create DataLoader
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels).long())
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels).long())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Early stopping setup
best_val_accuracy = 0.0
no_improvement_count = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} Training'):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation loop
    model.eval()
    all_val_preds = []
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation'):
            val_input_ids, val_attention_mask, val_labels = val_batch
            val_input_ids, val_attention_mask, val_labels = val_input_ids.to(device), val_attention_mask.to(device), val_labels.to(device)

            val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
            val_preds = torch.argmax(val_outputs.logits, dim=1).cpu().numpy()
            all_val_preds.extend(val_preds)

    # Calculate validation accuracy
    val_accuracy = accuracy_score(val_labels, all_val_preds)

    print(f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy}')

    # Check for improvement
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improvement_count = 0
        # Save model checkpoint
        torch.save(model.state_dict(), f'bert_sentiment_model_best.pt')
    else:
        no_improvement_count += 1

    # Check early stopping criteria
    if no_improvement_count >= patience:
        print(f'Early stopping after {patience} epochs without improvement.')
        break