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


def main():
    main_df = load_data(FILE_NAME)
    training_data, training_labels, test_data, test_labels = separate_data(main_df)


if __name__ == '__main__':
    main()
