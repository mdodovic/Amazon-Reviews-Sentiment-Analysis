from collections import Counter
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

file_path = 'dataset/finefoods.txt' 

# Read the dataset
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        user_ids = []
        for line in file:
            if line.startswith('review/userId:'):
                user_id = line.split('review/userId:')[1].strip()
                user_ids.append(user_id)
    return user_ids

# Define the path to your dataset file

# Get the user IDs from the dataset
user_ids = read_dataset(file_path)

# Count the number of reviews per user ID
user_review_counts = Counter(user_ids)

# Get the distribution
review_counts = list(user_review_counts.values())

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(review_counts, bins=range(1, max(review_counts)+1), align='left', color='blue', edgecolor='black')
plt.title('Number of Reviews per Reviewer')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.yscale('log')  # Use logarithmic scale for better visualization if needed
plt.grid(axis='y', alpha=0.75)
plt.savefig("users distribution.png", dpi = 90)
plt.show()
