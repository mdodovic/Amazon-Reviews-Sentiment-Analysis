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