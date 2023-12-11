import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset_wrapper import read_dataset, preprocess_for_bert

path_to_dataset = 'C:/Users/matij/Desktop/ReviewsML/dataset/text.txt'


def main():

    reviews, scores = read_dataset(path_to_dataset)
    sentiments, tokenized_reviews = preprocess_for_bert(reviews, scores)

    # Print the first 10 reviews
    for i  in range(len(reviews)):
        print(scores[i], ":", reviews[i], "->", sentiments[i])

    print("BERT!")
    # Load pre-trained BERT model and tokenizer
    # Check if the number of reviews and scores match
    if len(reviews) != len(scores):
        raise ValueError("Number of reviews and scores must be the same.")

    # Tokenize and encode the reviews
    input_ids = []
    attention_masks = []

    max_length = max(tensor['input_ids'].shape[1] for tensor in tokenized_reviews)

    for tokens in tokenized_reviews:
        padding_length = max_length - tokens['input_ids'].shape[1]
        input_ids.append(torch.nn.functional.pad(tokens['input_ids'], (0, padding_length), value=0))
        attention_masks.append(torch.nn.functional.pad(tokens['attention_mask'], (0, padding_length), value=0))

    # Stack the tensors
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    # Convert sentiments to numerical labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(sentiments)
    labels = torch.tensor(encoded_labels)

    # Load pre-trained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create a DataLoader for training
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop (you may need to adjust this based on your specific use case)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Function to predict sentiment
    def predict_sentiment(review):
        model.eval()
        encoded_review = tokenizer(review, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits).item()
        predicted_sentiment = label_encoder.classes_[predicted_label]
        return predicted_sentiment

    # Example usage for prediction
    sample_review = "This product is great! I love it."
    predicted_sentiment = predict_sentiment(sample_review)
    print(f"Predicted Sentiment: {predicted_sentiment}")

if __name__ == "__main__":
    main()
