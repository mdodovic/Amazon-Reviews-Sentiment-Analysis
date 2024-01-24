import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure the necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # If available, for WordNet lemmatization


def normalize_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Tokenize
    tokens = word_tokenize(text)

    # Filter out non-alphabetic tokens and stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

name_hash = {}
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
            # elif line.startswith('review/profileName'):
            #     if name_hash.has(line.split('review/profileName:')[1].strip()):
            #         name_hash[line.split('review/profileName:')[1].strip()] += 1
            #     else:
            #         name_hash[line.split('review/profileName:')[1].strip()] = 1
    return reviews, labels

# Function to preprocess and normalize reviews
def preprocess_reviews(reviews):
    return [normalize_text(review) for review in reviews]

def balance_dataset(reviews, labels, sampling_method='over'):
    labels_np = np.array(labels)
    resampled_indices = np.arange(len(labels)).reshape(-1, 1)

    if sampling_method == 'over':
        sampler = RandomOverSampler(random_state=42)
    elif sampling_method == 'under':
        sampler = RandomUnderSampler(random_state=42)
    else:
        # If no valid method specified, return the original dataset
        return reviews, labels_np

    # Apply sampling
    resampled_indices, _ = sampler.fit_resample(resampled_indices, labels_np)

    # Use the resampled indices to create the resampled reviews and labels
    reviews_resampled = [reviews[index[0]] for index in resampled_indices]
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
    plt.savefig("Dataset label distribution.png", dpi=90)
    plt.show()

# Function to plot review length distribution
def plot_review_length_distribution(reviews):
    review_lengths = [len(review.split()) for review in reviews]
    plt.figure(figsize=(8, 6))
    plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title('Review Length Distribution')
    plt.xlabel('Number of Words in Review')
    plt.ylabel('Frequency')
    plt.savefig("Dataset length distribution.png", dpi=90)
    plt.show()


def plot_review_length_distribution_better_scale(reviews):
    review_lengths = [len(review.split()) for review in reviews]
    plt.figure(figsize=(8, 6))
    plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black', log=True)  # Added log=True for logarithmic scale
    plt.title('Review Length Distribution')
    plt.xlabel('Number of Words in Review')
    plt.ylabel('Frequency (Log Scale)')  # Updated label to indicate log scale
    #plt.grid(True, which="both", ls="-")  # Add grid for better readability
    plt.grid(True, which="both", ls=":", color='gray', alpha=0.5)  # Lighter grid lines
    plt.savefig("Dataset_length_distribution_log_scale.png", dpi=90)
    plt.show()


# Plot name distribution 
# def plot_distribution_per_name(name_hash):

#     for key, value in name_hash.items():
#         print(key, value)

def fetch_data(path, sampling_method='over'):
    raw_reviews, raw_labels = read_dataset(path)
    #plot_label_distribution(raw_labels)
    #plot_review_length_distribution(raw_reviews)
    plot_review_length_distribution_better_scale(raw_reviews)

    # return raw_reviews, raw_labels
    #preprocessed_reviews = preprocess_reviews(raw_reviews)
    #balanced_reviews, balanced_labels = balance_dataset(preprocessed_reviews, raw_labels, sampling_method)

    # plot_label_distribution(raw_labels)
    # plot_review_length_distribution(raw_reviews)

    # plot_review_length_distribution(preprocessed_reviews)

    # plot_label_distribution(balanced_labels)
    # plot_review_length_distribution(balanced_reviews)

    # plot_distribution_per_name(name_hash)

    # return balanced_reviews, balanced_labels


#fetch_data('dataset/text.txt', sampling_method='under')
#fetch_data('dataset/text.txt', sampling_method='over')
#fetch_data('dataset/text.txt', sampling_method='over')
#fetch_data('dataset/finefoods.txt', sampling_method='under')
#fetch_data('dataset/finefoods.txt', sampling_method='over')
fetch_data('dataset/finefoods.txt', sampling_method='none')
