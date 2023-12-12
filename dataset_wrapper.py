from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


# Modify the read_dataset function in your dataset_wrapper.py to include label conversion
def read_dataset(path):
    reviews = []
    labels = []

    with open(path, 'r') as file:
        for line in file:
            if line.startswith('review/text:'):
                review = line.split('review/text:')[1].strip()
                reviews.append(review)
            elif line.startswith('review/score:'):
                score = float(line.split('review/score:')[1].strip())
                # Convert scores to labels
                if score < 3.0:
                    label = 0  # Negative
                elif score == 3.0:
                    label = 1  # Neutral
                else:
                    label = 2  # Positive
                labels.append(label)

    return reviews, labels


def preprocess_for_bert(reviews, scores):
    sentiments = []
    tokenized_reviews = []

    label_encoder = LabelEncoder()
    encoded_scores = label_encoder.fit_transform([0 if s <= 2 else 1 if s == 3 else 2 for s in scores])

    for review, encoded_score in zip(reviews, encoded_scores):
        sentiment = "negative" if encoded_score == 0 else "neutral" if encoded_score == 1 else "positive"
        sentiments.append(sentiment)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(review, padding=True, truncation=True, return_tensors='pt')
        tokenized_reviews.append(tokens)

    return sentiments, tokenized_reviews