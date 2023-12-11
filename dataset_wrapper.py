from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


def read_dataset(path_to_dataset: str):

    # Initialize lists to store scores and reviews
    scores = []
    reviews = []

    # Open the text file for reading
    with open(path_to_dataset, 'r') as file:
        lines = file.readlines()

    # Process the lines to extract scores and reviews
    current_score = None
    current_review = ""

    for line in lines:
        line = line.strip()
        if line.startswith("review/score:"):
            # Extract score
            current_score = float(line.split(":")[1].strip())
        elif line.startswith("review/text:"):
            # Extract review
            current_review = line.split(":")[1].strip()
            # Add the current score and review to the lists
            scores.append(current_score)
            reviews.append(current_review)

    return reviews, scores


def preprocess_for_bert(reviews, scores):
    sentiments = []
    tokenized_reviews = []

    label_encoder = LabelEncoder()
    encoded_scores = label_encoder.fit_transform([1 if s < 3 else 2 if s == 3 else 3 for s in scores])

    for review, encoded_score in zip(reviews, encoded_scores):
        sentiment = "positive" if encoded_score == 2 else "negative" if encoded_score == 0 else "neutral"
        sentiments.append(sentiment)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(review, padding=True, truncation=True, return_tensors='pt')
        tokenized_reviews.append(tokens)

    return sentiments, tokenized_reviews