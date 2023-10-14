#
#   Cian Herlihy - R00205604
#
#   Machine Learning Assignment 1
#

import pandas as pd

FILE_NAME = 'movie_reviews.xlsx'


def load_data(file):  # Task 1 part 1
    df = pd.read_excel(file)  # Reads Excel file and sets it to df (dataframe)
    return df


def separate_data(df):  # Task 1 part 2
    training_df = df[df['Split'] == 'train']
    test_df = df[df['Split'] == 'test']

    training_data = training_df['Review'].tolist()
    training_labels = training_df['Sentiment'].tolist()
    test_data = test_df['Review'].tolist()
    test_labels = test_df['Sentiment'].tolist()

    num_positive_training = training_labels.count('positive')
    num_negative_training = training_labels.count('negative')
    num_positive_test = test_labels.count('positive')
    num_negative_test = test_labels.count('negative')

    print(f"Training Data - Positive | Negative: {num_positive_training} | {num_negative_training}")
    print(f"Test Data     - Positive | Negative: {num_positive_test} | {num_negative_test}")

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


def calculate_priors_and_likelihoods(positive_reviews, negative_reviews):  # Task 4
    total_reviews = len(positive_reviews) + len(negative_reviews)
    prior_positive = len(positive_reviews) / total_reviews
    prior_negative = len(negative_reviews) / total_reviews

    alpha = 1  # Smoothing factor
    all_reviews = positive_reviews + negative_reviews
    unique_words = set(word for review in all_reviews for word in review.split())
    likelihoods = {}

    for word in unique_words:
        count_in_positive = sum(1 for review in positive_reviews if word in review) + alpha
        count_in_negative = sum(1 for review in negative_reviews if word in review) + alpha

        likelihood_positive = count_in_positive / (len(positive_reviews) + alpha * len(unique_words))
        likelihood_negative = count_in_negative / (len(negative_reviews) + alpha * len(unique_words))

        likelihoods[word] = (likelihood_positive, likelihood_negative)

    return prior_positive, prior_negative, likelihoods


def main():  # Main Function
    main_df = load_data(FILE_NAME)
    training_data, training_labels, test_data, test_labels = separate_data(main_df)
    filter_word_list = remove_special_chars(training_data, 3, 5)
    word_presence_dict = count_word_occurrences_in_reviews(training_data, filter_word_list)


if __name__ == '__main__':
    main()
