from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import re
import matplotlib.pyplot as plt


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


# Use the read_dataset function to get reviews and labels
reviews, labels = read_dataset('dataset/text.txt')  # Replace with the actual path of your dataset

# Count the frequency of each label
label_counts = [labels.count(0), labels.count(1), labels.count(2)]
label_names = ['Negative', 'Neutral', 'Positive']

# Plotting the label distribution
plt.figure(figsize=(8, 6))
plt.bar(label_names, label_counts, color=['red', 'blue', 'green'])
plt.title('Label Distribution in Dataset')
plt.xlabel('Sentiment Labels')
plt.ylabel('Frequency')
plt.show()

# Calculating review lengths
review_lengths = [len(review.split()) for review in reviews]

# Plotting review length distribution
plt.figure(figsize=(8, 6))
plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Review Length Distribution')
plt.xlabel('Number of Words in Review')
plt.ylabel('Frequency')
plt.show()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation (preserving intra-word dashes)
    text = re.sub(r'(?<!\w)-(?!\w)', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing to each review
preprocessed_reviews = [preprocess_text(review) for review in reviews]

# Calculating review lengths
review_lengths = [len(preprocessed_review.split()) for preprocessed_review in preprocessed_reviews]

# Plotting review length distribution
plt.figure(figsize=(8, 6))
plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Review Length Distribution')
plt.xlabel('Number of Words in Review')
plt.ylabel('Frequency')
plt.show()