from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


# Function to read the dataset
def read_dataset(path):
    reviews = []
    scores = []

    with open(path, 'r') as file:
        lines = file.readlines()

    current_score = None
    current_review = ""

    for line in lines:
        line = line.strip()
        if line.startswith("review/score:"):
            current_score = float(line.split(":")[1].strip())
            scores.append(current_score)
        elif line.startswith("review/text:"):
            current_review = line.split(":")[1].strip()
            reviews.append(current_review)

    return reviews, scores


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