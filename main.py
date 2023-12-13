path_to_dataset = 'C:/Users/matij/Desktop/ReviewsML/dataset/text.txt'

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


# Assuming the dataset_wrapper.py has been correctly imported
from dataset_wrapper import read_dataset

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

# Function to compute accuracy
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Read and preprocess the dataset
reviews, scores = read_dataset(path_to_dataset)  # Replace with your dataset path

# Splitting the dataset into training, validation, and test sets
train_reviews, temp_reviews, train_scores, temp_scores = train_test_split(reviews, scores, test_size=0.2, random_state=42)
val_reviews, test_reviews, val_scores, test_scores = train_test_split(temp_reviews, temp_scores, test_size=0.5, random_state=42)

# Tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset objects for training, validation, and testing
train_dataset = SentimentDataset(train_reviews, train_scores, tokenizer)
val_dataset = SentimentDataset(val_reviews, val_scores, tokenizer)
test_dataset = SentimentDataset(test_reviews, test_scores, tokenizer)

# BERT model for sequence classification with 3 classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train_dataset = train_dataset
trainer.eval_dataset = val_dataset
trainer.train()

# Evaluate the model on the test set	
test_results = trainer.evaluate(test_dataset)
print("Test Set Results:", test_results)