path_to_dataset = 'C:/Users/matij/Desktop/ReviewsML/dataset/text.txt'

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

# Read and preprocess the dataset
reviews, scores = read_dataset(path_to_dataset)  # Replace with your dataset path

# Tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Splitting the dataset into training and validation sets
train_reviews, val_reviews, train_scores, val_scores = train_test_split(reviews, scores, test_size=0.1)

# Create dataset objects for training and validation
train_dataset = SentimentDataset(train_reviews, train_scores, tokenizer)
val_dataset = SentimentDataset(val_reviews, val_scores, tokenizer)

# Load pre-trained BERT model for sequence classification
# Adjust num_labels to match the number of sentiment classes in your dataset
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
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model (optional)
evaluation_results = trainer.evaluate()
print(evaluation_results)
