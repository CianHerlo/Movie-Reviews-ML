#
#   Cian Herlihy - R00205604
#
#   Machine Learning Assignment 1
#

import pandas as pd

FILE_NAME = 'movie_reviews.xlsx'


def load_data(file):
    df = pd.read_excel(file)  # Reads Excel file and sets it to df (dataframe)
    return df


def separate_data(df):
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


def remove_special_chars(data_list, min_word_len, min_word_count):
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


def count_word_occurrences_in_reviews(review_set, selected_words):
    word_occurrence_count = {}
    for word in selected_words:
        word_occurrence_count[word] = 0

    for review in review_set:
        words_in_review = set(review.split())

        for word in selected_words:
            if word in words_in_review:
                word_occurrence_count[word] += 1

    return word_occurrence_count


def main():
    main_df = load_data(FILE_NAME)
    training_data, training_labels, test_data, test_labels = separate_data(main_df)
    filter_word_list = remove_special_chars(training_data, 3, 5)
    word_occurrence_count = count_word_occurrences_in_reviews(training_data, filter_word_list)


if __name__ == '__main__':
    main()
