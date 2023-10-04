import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

from dataloader import load_and_preprocess_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path_to_model = './models/bert_amazon_food_review222_drive'
path_to_file = "./dataset/" + "test.txt"

path = path_to_file
#path = './drive/MyDrive/TwitterSentimentAnalysis/finefoods.txt'
# Initialize empty lists to store scores and texts
scores = []
texts = []

# Open and read the file
with open(path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Initialize variables to store temporary data
current_score = None
current_text = []

# Process each line in the file
for line in lines:
    line = line.strip()

    # If the line starts with "review/score:", extract the score
    if line.startswith('review/score:'):
        current_score = float(line.split(': ')[1])
    # If the line starts with "review/text:", extract the text
    elif line.startswith('review/text:'):
        current_text.append(line.split(': ')[1])
    # If the line is empty, it indicates the end of a review block
    elif not line:
        # Join the lines of the review text and append to the texts list
        if current_score is not None and current_text:
            scores.append(current_score)
            texts.append(' '.join(current_text))

        # Reset variables for the next review
        current_score = None
        current_text = []

# Now you have the review scores in the 'scores' list and the review texts in the 'texts' list
# Printing the first 5 review scores and texts
for i in range(5):
    print(f"Review {i + 1} - Score: {scores[i]}, Text: {texts[i]}\n")
print(len(scores))
print(len(texts))


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the pretrained BERT model and tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model = model.to(device)


# Tokenize your text data and pad or truncate to a consistent length
max_length = 128  # Adjust the maximum sequence length as needed

tokenized_texts = [tokenizer(
    text,
    padding='max_length',  # Pad to the specified max length
    truncation=True,        # Truncate if the text exceeds max length
    max_length=max_length,
    return_tensors='pt'
) for text in texts]

# Convert sentiment scores to labels (assuming 6 sentiment classes)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(scores)

# Combine tokenized inputs into tensors
input_ids = torch.cat([text['input_ids'] for text in tokenized_texts], dim=0).to(device)
attention_mask = torch.cat([text['attention_mask'] for text in tokenized_texts], dim=0).to(device)
labels = torch.tensor(labels).to(device)

# Split the data into train and validation sets
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=420
)

# Create data loaders for training and validation data
batch_size = 64  # Adjust the batch size as needed
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# ... (previous code)

# Lists to store loss values and accuracy values
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
num_epochs = 10  # Adjust the number of epochs as needed
print_every = 100  # Print training loss every `print_every` steps

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training loop
    model.train()
    total_loss = 0
    correct_predictions_train = 0
    total_samples_train = 0

    for step, batch in enumerate(tqdm(train_data_loader, desc="Training")):
        optimizer.zero_grad()
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        correct_predictions_train += (predicted_labels == labels).sum().item()
        total_samples_train += len(labels)

        if (step + 1) % print_every == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Step {step + 1}/{len(train_data_loader)} - Loss: {avg_loss:.4f}")

    avg_train_loss = total_loss / len(train_data_loader)
    train_accuracy = correct_predictions_train / total_samples_train * 100
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Average training loss: {avg_train_loss:.4f} - Training accuracy: {train_accuracy:.2f}%")

    # Validation loop
    model.eval()
    val_loss = 0
    correct_predictions_val = 0
    total_samples_val = 0

    for batch in val_data_loader:
        with torch.no_grad():
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            val_loss += loss.item()

            # Calculate accuracy
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)
            correct_predictions_val += (predicted_labels == labels).sum().item()
            total_samples_val += len(labels)

    avg_val_loss = val_loss / len(val_data_loader)
    val_accuracy = correct_predictions_val / total_samples_val * 100
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Validation loss: {avg_val_loss:.4f} - Validation accuracy: {val_accuracy:.2f%}")

    # Save the fine-tuned model
    model.save_pretrained(path_to_model)

# Plotting
plt.figure(figsize=(12, 6))

# Plotting Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
