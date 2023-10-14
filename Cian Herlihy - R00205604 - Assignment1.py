#
#   Cian Herlihy - R00205604
#
#   Machine Learning Assignment 1
#

import pandas as pd
import math

FILE_NAME = 'movie_reviews.xlsx'


def load_excel_file(file):  # Task 1 part 1
    df = pd.read_excel(file)  # Reads Excel file and sets it to df (dataframe)
    return df


def separate_data(df):  # Task 1 part 2
    training_df = df[df['Split'] == 'train']
    test_df = df[df['Split'] == 'test']

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


def remove_special_chars(data_list, min_word_len, min_word_count):  # Task 2
    word_count_dict = {}

    for review in data_list:
        review = review.lower()
        words = review.split()

        for word in words:
            if len(word) >= min_word_len:
                if word in word_count_dict:
                    word_count_dict[word] += 1
                else:
                    word_count_dict[word] = 1

    filtered_words = []
    for word, count in word_count_dict.items():
        if count >= min_word_count:
            filtered_words.append(word)

    return filtered_words


def count_word_occurrences_in_reviews(review_set, selected_words):  # Task 3
    word_presence_dict = {}
    for word in selected_words:
        word_presence_dict[word] = 0

    for review in review_set:
        words_in_review = set(review.split())

        for word in selected_words:
            if word in words_in_review:
                word_presence_dict[word] += 1

    return word_presence_dict


def calculate_priors_and_likelihoods(words_dict, training_data):  # Task 4
    smooth_factor = 1
    likelihoods = {}
    pos_reviews = []
    neg_reviews = []
    word_list = list(words_dict.keys())
    for review in training_data:
        if review.endswith("positive"):
            pos_reviews.append(review)
        elif review.endswith("negative"):
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
        return "positive"
    else:
        return "negative"


def main():  # Main Function
    # Task 1
    main_df = load_excel_file(FILE_NAME)
    training_data, training_labels, test_data, test_labels = separate_data(main_df)
    # Task 2
    filter_word_list = remove_special_chars(training_data, 3, 5)
    # Task 3
    word_presence_dict = count_word_occurrences_in_reviews(training_data, filter_word_list)
    # Task 4
    prior_pos, prior_neg, likelihoods = calculate_priors_and_likelihoods(word_presence_dict, training_data)
    # Task 5
    predict_sentiment(new_review, prior_pos, prior_neg, likelihoods)
    # Task 6


if __name__ == '__main__':
    main()
