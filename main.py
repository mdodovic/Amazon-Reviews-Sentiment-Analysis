from dataset_wrapper import read_dataset, preprocess_for_bert

path_to_dataset = 'C:/Users/matij/Desktop/ReviewsML/dataset/text.txt'


def main():

    reviews, scores = read_dataset(path_to_dataset)
    sentiments, tokenized_reviews = preprocess_for_bert(reviews, scores)

    # Print the first 10 reviews
    for i  in range(len(reviews)):
        print(scores[i], sentiments[i])





if __name__ == "__main__":
    main()
