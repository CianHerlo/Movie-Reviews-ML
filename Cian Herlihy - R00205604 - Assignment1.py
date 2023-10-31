import re
import pandas as pd
import math
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#
#   Cian Herlihy - R00205604
#
#   Machine Learning Assignment 1
#

FILE_NAME = 'movie_reviews.xlsx'


def load_excel_file(file):
    print("Task 1: Loading Excel File")
    df = pd.read_excel(file)  # Reads Excel file and sets it to df (dataframe)
    return df  # Return Dataframe


def separate_data(df):
    training_df = df[df['Split'] == 'train']  # Training dataframe
    test_df = df[df['Split'] == 'test']  # Test dataframe

    training_data = training_df['Review'].tolist()  # Gather reviews into list for training
    training_labels = training_df['Sentiment'].tolist()  # Gather sentiments into list for training
    test_data = test_df['Review'].tolist()  # Gather reviews into list for test data
    test_labels = test_df['Sentiment'].tolist()  # Gather sentiments into list for test data

    num_pos_training = training_labels.count('positive')  # Get sum of positive reviews in training data
    num_neg_training = training_labels.count('negative')  # Get sum of negative reviews in training data
    num_pos_test = test_labels.count('positive')  # Get sum of positive reviews in test data
    num_neg_test = test_labels.count('negative')  # Get sum of negative reviews in test data

    print(
        f"Training Data - Positive | Negative: {num_pos_training} | {num_neg_training}")  # Print counts for training data
    print(f"Test Data     - Positive | Negative: {num_pos_test} | {num_neg_test}")  # Print counts for test data
    return training_data, training_labels, test_data, test_labels  # Return datasets


def count_positive_negative_reviews(df):
    print("Task 4: Count positive / negative reviews")
    sum_pos = df['Sentiment'].eq('positive').sum()  # Get sum of total positive reviews
    sum_neg = df['Sentiment'].eq('negative').sum()  # Get sum of total negative reviews
    return sum_pos, sum_neg  # Return sums


def filter_reviews(reviews, min_word_length, min_word_appearances):
    word_counts = {}  # Create dictionary for keeping word popularity
    filtered_words = []  # Create empty list for words being filtered
    for review in reviews:  # Loop through all reviews

        # https://stackoverflow.com/questions/6323296/python-remove-anything-that-is-not-a-letter-or-number
        # Use regex to get only Alpha-Numeric characters and make reviews lowercase
        # I had another way of doing this previously which was much slower but found this on StackOverflow
        cleaned_review = re.sub(r'[^a-zA-Z0-9 ]', '', review.lower())

        words = cleaned_review.split()  # Split review into individual words
        for word in words:  # Loop through each word in reviews
            if len(word) >= min_word_length:  # Check if word meets minimum length requirements
                if word in word_counts:  # If word is already added to dictionary
                    word_counts[word] += 1  # Increment by 1
                else:  # Word does not appear in dictionary
                    word_counts[word] = 1  # Add word to dictionary and Set Value to 1

    for word, count in word_counts.items():  # Word & Count is iterated through in a loop of all word_counts key-value pairs
        if count >= min_word_appearances:  # If word appearances is >= minimum word count param
            filtered_words.append(word)  # Add word to filtered_words list
    return filtered_words  # Return list of filtered words


def featured_word_count_in_reviews(review_data, review_labels, filtered_words):
    feature_count_pos_review = {}  # Dictionary for positive review appearances for featured words
    feature_count_neg_review = {}  # Dictionary for negative review appearances for featured words
    total_feature_word_count = {}  # Create an empty dictionary to store word counts
    for word in filtered_words:  # Loop through all filtered words
        total_feature_word_count[
            word] = 0  # Add filtered words as key to dictionary and set value to 0 to initialise the dictionary

    for i, review in enumerate(
            review_data):  # For loop that will iterate through each review and i to keep count of index
        words_in_review = review.split()  # Split review into individual words
        label = review_labels[i]  # Retrieve sentiment of review

        if label == "positive":  # If review is positive
            word_counts = feature_count_pos_review  # Set word_counts as positive dictionary
        else:  # If Review is Negative
            word_counts = feature_count_neg_review  # Set word_counts as negative dictionary

        for word in words_in_review:  # Loop through each word in review
            if word in filtered_words:  # If word is in the filtered words list
                if word in word_counts:  # If it exists in word_counts dictionary
                    word_counts[word] += 1  # Increment by 1
                else:  # If word is not in dictionary yet
                    word_counts[word] = 1  # Add to dictionary and set value to 1

                if word in total_feature_word_count:  # If word is in the dictionary total_feature_word_count
                    total_feature_word_count[word] += 1  # Increment word count by 1

    return feature_count_pos_review, feature_count_neg_review, total_feature_word_count  # Return counts


def calculate_likelihoods(word_list, pos_counts, neg_counts, alpha):
    likelihoods = {}  # Create dictionary for likelihoods
    for word in word_list:  # Loop through words in filtered words list
        total_word_count = pos_counts.get(word, 0) + neg_counts.get(word, 0)  # Get total count with pos + neg
        if total_word_count == 0:  # If total is 0
            pos_likelihood = neg_likelihood = 0  # Set positive & negative to 0 likelihood
        else:  # Total more than 0
            pos_likelihood = (pos_counts.get(word, 0) + alpha) / (
                        total_word_count + 2 * alpha)  # Calculate likelihood for positive sentiment with smoothing factor
            neg_likelihood = (neg_counts.get(word, 0) + alpha) / (
                        total_word_count + 2 * alpha)  # Calculate likelihood for negative sentiment with smoothing factor
        likelihoods[word] = (
        pos_likelihood, neg_likelihood)  # Set value for word with a tuple of positive vs negative sentiment likelihood

    print(f"Likelihoods: {likelihoods}")  # Print likelihoods
    return likelihoods  # Return likelihoods dictionary


def calculate_priors(total_pos_reviews, total_neg_reviews, alpha):
    prior_pos = (total_pos_reviews + alpha) / (
                total_pos_reviews + total_neg_reviews + 2 * alpha)  # Calculate prior for positive sentiment with smoothing factor
    prior_neg = (total_neg_reviews + alpha) / (
                total_pos_reviews + total_neg_reviews + 2 * alpha)  # Calculate prior for negative sentiment with smoothing factor
    print(f"Prior Positive Reviews: {prior_pos}")  # Print prior positive
    print(f"Prior Negative Reviews: {prior_neg}")  # Print prior negative
    return prior_pos, prior_neg  # Return priors


def predict_sentiment(new_review, prior_pos, prior_neg, likelihoods):
    log_prior_pos = math.log(prior_pos)  # Calculates logarithm for prior positive
    log_prior_neg = math.log(prior_neg)  # Calculates logarithm for prior negative
    words = new_review.split()  # Split review into words
    log_likelihood_pos = 0  # Initialise var
    log_likelihood_neg = 0  # Initialise var

    for word in words:  # Loop through all words in review
        if word in likelihoods:  # If word is in dictionary of likelihoods
            log_likelihood_pos += math.log(likelihoods[word][0])  # Add logarithm of positive sentiment likelihood
            log_likelihood_neg += math.log(likelihoods[word][1])  # Add logarithm of negative sentiment likelihood

    log_posterior_pos = log_prior_pos + log_likelihood_pos  # Sum priors and likelihoods for positive sentiments
    log_posterior_neg = log_prior_neg + log_likelihood_neg  # Sum priors and likelihoods for negative sentiments

    if log_posterior_pos > log_posterior_neg:  # If positive is more than negative sentiment
        prediction = "positive"  # Predict Positive
    else:  # If positive is equal or less than negative sentiment
        prediction = "negative"  # Predict Negative
    return prediction


def kfold_cross_validation(training_data, training_labels, prior_pos, prior_neg, likelihoods):
    predictions = []            # Empty List to add predicted sentiments
    actual_sentiments = []      # Actual results in dataset
    kf = KFold(n_splits=5)      # Create KFold with 5 splits
    for train_index, val_index in kf.split(training_data):  # Loop through training data
        x_train_fold, x_val_fold = training_data.iloc[train_index], training_data.iloc[val_index]       # Splits data into training|validating
        y_train_fold, y_val_fold = training_labels.iloc[train_index], training_labels.iloc[val_index]   # Splits data into corresponding training|validating

        for review in x_val_fold:                               # Cycle through reviews
            prediction = predict_sentiment(str(review), prior_pos, prior_neg, likelihoods)  # Predict reviews sentiment
            predictions.append(prediction)                      # Add prediction to list
        actual_sentiments.extend(y_val_fold)                    # Add actual sentiment to list
    accuracy = accuracy_score(actual_sentiments, predictions)   # Calculate accuracy of model
    print("Accuracy: {:.2f}%".format(accuracy * 100))           # Print accuracy. 50% being worst accuracy


def main():  # Main Function
    # Task 1
    print("Task 1: Split data")
    main_df = load_excel_file(FILE_NAME)
    training_data, training_labels, test_data, test_labels = separate_data(main_df)

    # Task 2
    print("Task 2: Filter Words from Reviews")
    filter_word_list = filter_reviews(training_data, 8, 100)

    # Task 3
    print("Task 3: Count Featured Words in Reviews")
    feature_count_pos_review, feature_count_neg_review, total_feature_word_count = featured_word_count_in_reviews(
        training_data, training_labels, filter_word_list)

    # Task 4
    print("Task 4: Calculate Likelihoods")
    alpha = 1  # alpha = smoothing factor
    likelihoods = calculate_likelihoods(filter_word_list, feature_count_pos_review, feature_count_neg_review, alpha)
    sum_pos, sum_neg = count_positive_negative_reviews(main_df)
    print("Task 4: Calculate Priors")
    prior_pos, prior_neg = calculate_priors(sum_pos, sum_neg, alpha)

    # Task 5
    print("Task 5: Predict Custom Review")
    prediction = predict_sentiment(
        "This movie was fantastic, exhilarating, and tremendously choreographed. Close to perfection",
        prior_pos, prior_neg, likelihoods)
    print(prediction)

    # Task 6
    print("Task 6: Cross Validation")
    kfold_cross_validation(pd.DataFrame(training_data), pd.DataFrame(training_labels), prior_pos, prior_neg, likelihoods)


if __name__ == '__main__':
    main()
