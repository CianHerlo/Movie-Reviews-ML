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


def count_positive_negative_reviews(df):
    sum_pos = df['Sentiment'].eq('positive').sum()
    sum_neg = df['Sentiment'].eq('negative').sum()
    return sum_pos, sum_neg


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


def featured_word_count_in_reviews(review_data, review_labels, filtered_words):  # Task 3
    word_counts_positive = {}
    word_counts_negative = {}

    for i, review in enumerate(review_data):
        words_in_review = review.split()
        label = review_labels[i]

        if label == "positive":
            word_counts = word_counts_positive
        else:
            word_counts = word_counts_negative

        for word in words_in_review:
            if word in filtered_words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

    word_occurrence_count = {}
    for word in filtered_words:
        word_occurrence_count[word] = 0

    for word in filtered_words:
        for review in review_data:
            if word in review:
                word_occurrence_count[word] += 1

    return word_counts_positive, word_counts_negative, word_occurrence_count


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


def calculate_likelihoods(word_list, pos_counts, neg_counts):  # Task 4 part 1
    likelihoods = {}

    for word in word_list:
        total_word_count = pos_counts.get(word, 0) + neg_counts.get(word, 0)
        if total_word_count == 0:
            pos_likelihood = neg_likelihood = 0
        else:
            pos_likelihood = pos_counts.get(word, 0) / total_word_count
            neg_likelihood = neg_counts.get(word, 0) / total_word_count
        likelihoods[word] = (pos_likelihood, neg_likelihood)

    return likelihoods


def calculate_priors(total_pos_reviews, total_neg_reviews):  # Task 4 part 2
    prior_pos = total_pos_reviews / (total_pos_reviews + total_neg_reviews)
    prior_neg = total_neg_reviews / (total_pos_reviews + total_neg_reviews)
    return prior_pos, prior_neg


def main():  # Main Function
    # Task 1
    main_df = load_excel_file(FILE_NAME)
    training_data, training_labels, test_data, test_labels = separate_data(main_df)
    # Task 2
    filter_word_list = filter_reviews(training_data, 8, 100)
    # Task 3
    word_counts_positive, word_counts_negative, word_presence_dict = featured_word_count_in_reviews(
            training_data, training_labels, filter_word_list)
    # Task 4
    likelihoods = calculate_likelihoods(filter_word_list, word_counts_positive, word_counts_negative)
    sum_pos, sum_neg = count_positive_negative_reviews(main_df)
    prior_pos, prior_neg = calculate_priors(sum_pos, sum_neg)
    print(f"Likelihoods: {likelihoods}")
    print(f"Prior Positive Reviews: {prior_pos}")
    print(f"Prior Negative Reviews: {prior_neg}")
    # Task 5

    # Task 6


if __name__ == '__main__':
    main()
