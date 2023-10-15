#
#   Cian Herlihy - R00205604
#
#   Machine Learning Assignment 1
#
import re
import pandas as pd
import math
from sklearn import model_selection
from sklearn import metrics
from sklearn import neighbors
import matplotlib.pyplot as plt

FILE_NAME = 'movie_reviews.xlsx'


def load_excel_file(file):  # Task 1 part 1
    df = pd.read_excel(file)  # Reads Excel file and sets it to df (dataframe)
    return df  # Return Dataframe


def separate_data(df):  # Task 1 part 2
    training_df = df[df['Split'] == 'train']  # Training Dataframe
    test_df = df[df['Split'] == 'test']  # Test Dataframe

    training_data = training_df['Review'].tolist()
    training_labels = training_df['Sentiment'].tolist()
    test_data = test_df['Review'].tolist()
    test_labels = test_df['Sentiment'].tolist()

    num_pos_training = training_labels.count('positive')
    num_neg_training = training_labels.count('negative')
    num_pos_test = test_labels.count('positive')
    num_neg_test = test_labels.count('negative')

    print(f"Training Data - Positive | Negative: {num_pos_training} | {num_neg_training}")
    print(f"Test Data     - Positive | Negative: {num_pos_test} | {num_neg_test}")

    return training_data, training_labels, test_data, test_labels


def filter_reviews(reviews, min_word_length, min_word_appearances):
    # Create a dictionary to keep track of word counts
    word_counts = {}

    # Process each review in the list
    for review in reviews:
        # Make the review lowercase and remove non-alphanumeric characters
        # https://stackoverflow.com/questions/6323296/python-remove-anything-that-is-not-a-letter-or-number
        cleaned_review = re.sub(r'[^a-zA-Z0-9 ]', '', review.lower())

        # Split the cleaned review into words
        words = cleaned_review.split()

        # Iterate through the words in the review
        for word in words:
            # Check if the word meets the minimum length requirement
            if len(word) >= min_word_length:
                # Update word counts in the dictionary
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

    # Create a list of filtered words that meet the minimum appearance requirement
    filtered_words = []
    for word, count in word_counts.items():
        if count >= min_word_appearances:
            filtered_words.append(word)
    return filtered_words


def count_word_occurrences_in_reviews(review_set, review_labels, filtered_words):  # Task 3
    word_presence_dict = {}
    for word in filtered_words:
        word_presence_dict[word] = 0

    for review in review_set:
        words_in_review = set(review.split())

        for word in filtered_words:
            if word in words_in_review:
                word_presence_dict[word] += 1

    print(word_presence_dict)
    return word_presence_dict


def calculate_priors_and_likelihoods(words_dict, training_data, training_labels):  # Task 4
    smooth_factor = 1
    likelihoods = {}
    pos_reviews = []
    neg_reviews = []
    word_list = list(words_dict.keys())
    for i, review in enumerate(training_data):
        if training_labels[i] == "positive":
            pos_reviews.append(review)
        elif training_labels[i] == "negative":
            neg_reviews.append(review)

    total_reviews = len(pos_reviews) + len(neg_reviews)
    prior_pos = len(pos_reviews) / total_reviews
    prior_neg = len(neg_reviews) / total_reviews

    for word in word_list:
        count_pos = pos_reviews.count(word) + smooth_factor
        count_neg = neg_reviews.count(word) + smooth_factor
        likely_pos = count_pos / (len(pos_reviews) + smooth_factor * len(word_list))
        likely_neg = count_neg / (len(neg_reviews) + smooth_factor * len(word_list))
        likelihoods[word] = (likely_pos, likely_neg)

    return prior_pos, prior_neg, likelihoods


def predict_sentiment(new_review, prior_pos, prior_neg, likelihoods):  # Task 5
    log_prior_pos = math.log(prior_pos)
    log_prior_neg = math.log(prior_neg)
    words = new_review.split()
    log_likelihood_pos = 0
    log_likelihood_neg = 0

    for word in words:
        if word in likelihoods:
            log_likelihood_pos += math.log(likelihoods[word][0])
            log_likelihood_neg += math.log(likelihoods[word][1])

    log_posterior_pos = log_prior_pos + log_likelihood_pos
    log_posterior_neg = log_prior_neg + log_likelihood_neg
    if log_posterior_pos > log_posterior_neg:
        prediction = "positive"
    else:
        prediction = "negative"

    return prediction


def k_fold_cross_validation(classifier, data, labels, k):  # Task 6 part 1
    accuracies = []
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

    for train_index, eval_index in kf.split(data):
        train_data, eval_data = [data[i] for i in train_index], [data[i] for i in eval_index]
        train_labels, eval_labels = [labels[i] for i in train_index], [labels[i] for i in eval_index]

        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(eval_data)

        accuracy = metrics.accuracy_score(eval_labels, predictions)
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_accuracy


def find_optimal_word_length_parameter(classifier, data, labels, k):  # Task 6 part 2
    optimal_length = 1
    max_mean_accuracy = 0
    for length in range(1, 11):
        word_list = extract_words(data, length)
        mean_accuracy = k_fold_cross_validation(classifier, word_list, labels, k)

        if mean_accuracy > max_mean_accuracy:
            max_mean_accuracy = mean_accuracy
            optimal_length = length

    return optimal_length


def evaluate_classifier_on_test_set(classifier, data, labels, test_data, test_labels, optimal_length):  # Task 6 part 3
    word_list = extract_words(data, optimal_length)
    classifier.fit(word_list, labels)
    predictions = classifier.predict(test_data)

    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
    tp = confusion_matrix[0, 0]
    tn = confusion_matrix[1, 1]
    fp = confusion_matrix[1, 0]
    fn = confusion_matrix[0, 1]

    accuracy = metrics.accuracy_score(test_labels, predictions)
    return confusion_matrix, tp, tn, fp, fn, accuracy


def main():  # Main Function
    # Task 1
    main_df = load_excel_file(FILE_NAME)
    training_data, training_labels, test_data, test_labels = separate_data(main_df)
    # Task 2
    filter_word_list = filter_reviews(training_data, 6, 50)
    # Task 3
    word_presence_dict = count_word_occurrences_in_reviews(training_data, training_labels, filter_word_list)
    # Task 4
    prior_pos, prior_neg, likelihoods = calculate_priors_and_likelihoods(word_presence_dict, training_data,
                                                                         training_labels)
    # Task 5
    predicted_sentiment = predict_sentiment("This movie is terrible",
                                            prior_pos, prior_neg, likelihoods)
    print(f"Predicted Sentiment: {predicted_sentiment}")
    # Task 6


if __name__ == '__main__':
    main()
