def load_and_preprocess_data(path_to_file, print_dataset_info=True):
    # Initialize empty lists to store scores and texts
    scores = []
    texts = []

    # Open and read the file
    with open(path_to_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize variables to store temporary data
    current_score = None
    current_text = []

    # Process each line in the file
    for line in lines:
        line = line.strip()

        # If the line starts with "review/score:", extract the score
        if line.startswith('review/score:'):
            current_score = float(line.split(': ')[1])
        # If the line starts with "review/text:", extract the text
        elif line.startswith('review/text:'):
            current_text.append(line.split(': ')[1])
        # If the line is empty, it indicates the end of a review block
        elif not line:
            # Join the lines of the review text and append to the texts list
            if current_score is not None and current_text:
                scores.append(current_score - 1) # Adjusted to be 0 to 4
                texts.append(' '.join(current_text))

            # Reset variables for the next review
            current_score = None
            current_text = []

    # Now you have the review scores in the 'scores' list and the review texts in the 'texts' list
    # Printing the first 5 review scores and texts
    if print_dataset_info:
        for i in range(5):
            print(f"Review {i + 1} - Score: {scores[i]}, Text: {texts[i]}\n")
        print(len(scores))
        print(len(texts))

    return texts, scores


def load_and_preprocess_data_no_subtract(path_to_file, print_dataset_info=True):

    scores = []
    texts = []

    # Open and read the file
    with open(path_to_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize variables to store temporary data
    current_score = None
    current_text = []

    # Process each line in the file
    for line in lines:
        line = line.strip()

        # If the line starts with "review/score:", extract the score
        if line.startswith('review/score:'):
            current_score = float(line.split(': ')[1])
        # If the line starts with "review/text:", extract the text
        elif line.startswith('review/text:'):
            current_text.append(line.split(': ')[1])
        # If the line is empty, it indicates the end of a review block
        elif not line:
            # Join the lines of the review text and append to the texts list
            if current_score is not None and current_text:
                scores.append(current_score)
                texts.append(' '.join(current_text))

            # Reset variables for the next review
            current_score = None
            current_text = []

    # Now you have the review scores in the 'scores' list and the review texts in the 'texts' list
    # Printing the first 5 review scores and texts
    for i in range(5):
        print(f"Review {i + 1} - Score: {scores[i]}, Text: {texts[i]}\n")
    print(len(scores))
    print(len(texts))


    return texts, scores