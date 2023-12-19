import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer

# Ensure the necessary NLTK data is downloaded
# nltk.download('stopwords')
# nltk.download('punkt')

# Function to normalize text
def normalize_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Function to read dataset from file
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
                label = 0 if score < 3.0 else (1 if score == 3.0 else 2)
                labels.append(label)
    return reviews, labels

# Function to preprocess and normalize reviews
def preprocess_reviews(reviews):
    return [normalize_text(review) for review in reviews]

def balance_dataset(reviews, labels):
    # Convert labels to NumPy array for oversampling
    labels_np = np.array(labels)

    # Apply over-sampling
    ros = RandomOverSampler(random_state=42)
    resampled_indices, _ = ros.fit_resample(np.arange(len(labels)).reshape(-1, 1), labels_np)

    # Use the resampled indices to create the resampled reviews
    reviews_resampled = [reviews[index] for index in resampled_indices.flatten()]

    # Extract the corresponding labels
    labels_resampled = labels_np[resampled_indices.flatten()]

    return reviews_resampled, labels_resampled


# Function to plot label distribution
def plot_label_distribution(labels):
    label_counts = np.unique(labels, return_counts=True)
    label_names = ['Negative', 'Neutral', 'Positive']
    plt.figure(figsize=(8, 6))
    plt.bar(label_names, label_counts[1], color=['red', 'blue', 'green'])
    plt.title('Label Distribution in Dataset')
    plt.xlabel('Sentiment Labels')
    plt.ylabel('Frequency')
    plt.show()

# Function to plot review length distribution
def plot_review_length_distribution(reviews):
    review_lengths = [len(review.split()) for review in reviews]
    plt.figure(figsize=(8, 6))
    plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title('Review Length Distribution')
    plt.xlabel('Number of Words in Review')
    plt.ylabel('Frequency')
    plt.show()


def fetch_data(path):
    raw_reviews, raw_labels = read_dataset(path)  
    # preprocessed_reviews = preprocess_reviews(raw_reviews)
    balanced_reviews, balanced_labels = balance_dataset(raw_reviews, raw_labels)

    plot_label_distribution(raw_labels)
    plot_review_length_distribution(raw_reviews)
    plot_label_distribution(balanced_labels)
    plot_review_length_distribution(balanced_reviews)

fetch_data('dataset/text.txt')