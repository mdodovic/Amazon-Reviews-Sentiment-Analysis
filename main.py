from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Assuming the dataset_wrapper.py has been correctly imported
from dataset_wrapper import read_dataset


device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
print("Using device:", device)

# Custom dataset class for BERT
class SentimentDataset(Dataset):
    def __init__(self, reviews, scores, tokenizer, max_len=512):
        self.reviews = reviews
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        score = self.scores[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.long)
        }

class FineTuningConfig:
    def __init__(self):
        # Path to Dataset
        self.dataset_path = 'dataset/finefoods.txt' 
        self.labels_num = 3 # Number of labels in the dataset

        # Model Training Parameters
        self.num_epochs = 3 # Number of training epochs
        self.train_batch_size = 32 # Batch size for training
        self.eval_batch_size = 32 # Batch size for evaluation
        self.learning_rate = 5e-5 # Learning rate for the optimizer
        self.warmup_steps = 500 # Number of warmup steps for learning rate scheduler
        self.weight_decay = 0.01 # Regularization parameter
        self.max_seq_length = 512 # Max length of input sequences

        # Strategy and steps
        self.log_and_eval_strategy = 'steps'
        self.log_and_eval_steps = 500  # Choose a value that suits your training regimen
        
        self.save_strategy = 'steps'
        self.save_steps = 500  # Choose a value that suits your training regimen
        self.save_total_limit = 1 

        # File Paths and Directories
        self.output_dir = './results' # Directory to save the model
        self.logging_dir = './logs'   # Directory to save logs

        # Dataset Splitting Parameters
        self.validation_split = 0.2  # Portion of the data for validation
        self.test_split = 0.5        # Portion of the validation data for testing
        self.random_state = 42       # Random state for reproducibility


# Function to compute accuracy
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

config = FineTuningConfig()

# Read and preprocess the dataset
reviews, scores = read_dataset(config.dataset_path)

# Splitting the dataset
train_reviews, temp_reviews, train_scores, temp_scores = train_test_split(
    reviews, scores, test_size=config.validation_split, random_state=config.random_state
)
val_reviews, test_reviews, val_scores, test_scores = train_test_split(
    temp_reviews, temp_scores, test_size=config.test_split, random_state=config.random_state
)

# Tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset objects for training, validation, and testing
train_dataset = SentimentDataset(train_reviews, train_scores, tokenizer, max_len=config.max_seq_length)
val_dataset = SentimentDataset(val_reviews, val_scores, tokenizer, max_len=config.max_seq_length)
test_dataset = SentimentDataset(test_reviews, test_scores, tokenizer, max_len=config.max_seq_length)

# BERT model for sequence classification with 3 classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.labels_num)
model.to(device)

# Update the TrainingArguments with the new strategies
training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    learning_rate=config.learning_rate,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    logging_dir=config.logging_dir,
    evaluation_strategy=config.log_and_eval_strategy,
    eval_steps=config.log_and_eval_steps,
    logging_strategy=config.log_and_eval_strategy,
    logging_steps=config.log_and_eval_steps,
    save_strategy=config.save_strategy,
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set	
test_results = trainer.evaluate(test_dataset)
print("Test Set Results:", test_results)

# Extracting the training history
history = trainer.state.log_history

# Initializing lists to store metrics
train_loss_set = []
valid_loss_set = []
train_accuracy_set = []
valid_accuracy_set = []

# Extracting loss and accuracy for each epoch
for entry in history:
    if 'loss' in entry:
        train_loss_set.append(entry['loss'])
    elif 'eval_loss' in entry:
        valid_loss_set.append(entry['eval_loss'])
    if 'eval_accuracy' in entry:
        valid_accuracy_set.append(entry['eval_accuracy'])

# Plotting training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_set, label='Training loss')
plt.plot(valid_loss_set, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy (if recorded)
if valid_accuracy_set:
    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracy_set, label='Validation accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.savefig('loss_accuracy.png', dpi=90)
plt.show()