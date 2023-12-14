from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Assuming the dataset_wrapper.py has been correctly imported
from dataset_wrapper import read_dataset


device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
print("Using device:", device)

# Custom callback for training accuracy
class TrainingAccuracyCallback(TrainerCallback):
    def __init__(self, train_dataset, batch_size):
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.device = device

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            model.eval()  # Set model to evaluation mode
            total_eval_accuracy = 0
            total_eval_steps = 0

            for batch in tqdm(DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False), desc="Accuracy Calculation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                total_eval_accuracy += (preds == batch["labels"]).sum().item()
                total_eval_steps += batch["labels"].size(0)

            avg_train_accuracy = total_eval_accuracy / total_eval_steps
            print(f"Training accuracy (step {state.global_step}): {avg_train_accuracy:.4f}")

            # Directly append the accuracy to the trainer's log history
            state.log_history.append({
                "step": state.global_step,
                "train_accuracy": avg_train_accuracy,
            })

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
        self.dataset_path = 'dataset/text.txt' 
        self.labels_num = 3 # Number of labels in the dataset

        # Model Training Parameters
        self.num_epochs = 3 # Number of training epochs
        self.train_batch_size = 32 # Batch size for training
        self.eval_batch_size = 32 # Batch size for evaluation
        self.learning_rate = 5e-5 # Learning rate for the optimizer
        self.warmup_steps = 1 # Number of warmup steps for learning rate scheduler
        self.weight_decay = 0.01 # Regularization parameter
        self.max_seq_length = 512 # Max length of input sequences

        # Strategy and steps
        self.log_and_eval_strategy = 'steps'
        self.log_and_eval_steps = 1  # Choose a value that suits your training regimen
        
        self.save_strategy = 'steps'
        self.save_steps = 1  # Choose a value that suits your training regimen
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

# Instantiate the training accuracy callback
train_acc_callback = TrainingAccuracyCallback(train_dataset, config.train_batch_size)

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
    compute_metrics=compute_metrics,
    callbacks=[train_acc_callback]
)

# Train the model
trainer.train()

# Evaluate the model on the test set	
test_results = trainer.evaluate(test_dataset)
print("Test Set Results:", test_results)

# Extracting the training history for plotting
train_loss_set = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
valid_loss_set = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
train_accuracy_set = [log['train_accuracy'] for log in trainer.state.log_history if 'train_accuracy' in log]
valid_accuracy_set = [log['eval_accuracy'] for log in trainer.state.log_history if 'eval_accuracy' in log]

# Plotting training and validation loss and accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss_set, label='Training Loss')
plt.plot(valid_loss_set, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_set, label='Training Accuracy')
plt.plot(valid_accuracy_set, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_metrics.png', dpi=100)
plt.show()