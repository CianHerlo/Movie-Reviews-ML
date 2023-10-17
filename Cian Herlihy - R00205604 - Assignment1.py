import re
import pandas as pd
import math
from sklearn import model_selection
from sklearn import metrics
from sklearn import neighbors
import matplotlib.pyplot as plt

#
#   Cian Herlihy - R00205604
#
#   Machine Learning Assignment 1
#

FILE_NAME = 'movie_reviews.xlsx'


def load_excel_file(file):
    print("Task 1: Loading Excel File")
    df = pd.read_excel(file)    # Reads Excel file and sets it to df (dataframe)
    return df                   # Return Dataframe


def separate_data(df):
    print("Task 1: Split data")
    training_df = df[df['Split'] == 'train']                # Training Dataframe
    test_df = df[df['Split'] == 'test']                     # Test Dataframe

    training_data = training_df['Review'].tolist()          # Gather Reviews into List for Training
    training_labels = training_df['Sentiment'].tolist()     # Gather Sentiments into List for Training
    test_data = test_df['Review'].tolist()                  # Gather Reviews into List for Test Data
    test_labels = test_df['Sentiment'].tolist()             # Gather Sentiments into List for Test Data

    num_pos_training = training_labels.count('positive')    # Get Sum of Positive Reviews in Training Data
    num_neg_training = training_labels.count('negative')    # Get Sum of Negative Reviews in Training Data
    num_pos_test = test_labels.count('positive')            # Get Sum of Positive Reviews in Test Data
    num_neg_test = test_labels.count('negative')            # Get Sum of Negative Reviews in Test Data

    print(f"Training Data - Positive | Negative: {num_pos_training} | {num_neg_training}")  # Print Counts for Training Data
    print(f"Test Data     - Positive | Negative: {num_pos_test} | {num_neg_test}")          # Print Counts for Test Data

    return training_data, training_labels, test_data, test_labels   # Return Datasets


def count_positive_negative_reviews(df):
    print("Task 4: Count positive / negative reviews")
    sum_pos = df['Sentiment'].eq('positive').sum()  # Get Sum of Total Positive Reviews
    sum_neg = df['Sentiment'].eq('negative').sum()  # Get Sum of Total Negative Reviews
    return sum_pos, sum_neg                         # Return Sums


def filter_reviews(reviews, min_word_length, min_word_appearances):
    print("Task 2: Filter Words from Reviews")
    word_counts = {}        # Create Dictionary for Keeping Word Popularity
    filtered_words = []     # Create Empty List for Words Being Filtered
    for review in reviews:  # Loop through all Reviews

        # https://stackoverflow.com/questions/6323296/python-remove-anything-that-is-not-a-letter-or-number
        # Use Regex to get only Alpha-Numeric Characters and Make Reviews Lowercase
        # I had another way of doing this previously which was much slower but found this on StackOverflow
        cleaned_review = re.sub(r'[^a-zA-Z0-9 ]', '', review.lower())

        words = cleaned_review.split()          # Split Review into Individual Words
        for word in words:                      # Loop Through each Word in Reviews
            if len(word) >= min_word_length:    # Check if Word meets Minimum Length Requirements
                if word in word_counts:         # If Word is already added to Dictionary
                    word_counts[word] += 1      # Increment by 1
                else:                           # Word Does Not Appear in Dictionary
                    word_counts[word] = 1       # Add Word to Dictionary and Set Value to 1

    for word, count in word_counts.items():     # Word & Count is iterated through in a loop of all word_counts key-value pairs
        if count >= min_word_appearances:       # If Word Appearances is >= Minimum Word Count Param
            filtered_words.append(word)         # Add Word to filtered_words List
    return filtered_words                       # Return List of Filtered Words


def featured_word_count_in_reviews(review_data, review_labels, filtered_words):
    print("Task 3: Count Featured Words in Reviews")
    word_counts_positive = {}
    word_counts_negative = {}
    word_occurrence_count = {word: 0 for word in filtered_words}

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

                if word in word_occurrence_count:
                    word_occurrence_count[word] += 1

    return word_counts_positive, word_counts_negative, word_occurrence_count


def calculate_likelihoods(word_list, pos_counts, neg_counts):
    print("Task 4: Calculate Likelihoods")
    likelihoods = {}
    for word in word_list:
        total_word_count = pos_counts.get(word, 0) + neg_counts.get(word, 0)
        if total_word_count == 0:
            pos_likelihood = neg_likelihood = 0
        else:
            pos_likelihood = pos_counts.get(word, 0) / total_word_count
            neg_likelihood = neg_counts.get(word, 0) / total_word_count
        likelihoods[word] = (pos_likelihood, neg_likelihood)

    print(f"Likelihoods: {likelihoods}")
    return likelihoods


def calculate_priors(total_pos_reviews, total_neg_reviews):
    print("Task 4: Calculate Priors")
    prior_pos = total_pos_reviews / (total_pos_reviews + total_neg_reviews)
    prior_neg = total_neg_reviews / (total_pos_reviews + total_neg_reviews)
    print(f"Prior Positive Reviews: {prior_pos}")
    print(f"Prior Negative Reviews: {prior_neg}")
    return prior_pos, prior_neg


def predict_sentiment(new_review, prior_pos, prior_neg, likelihoods):
    print("Task 5: Predict Custom Review")
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
    print(prediction)
    return prediction


def k_fold_cross_validation(classifier, data, labels, k):
    accuracies = []
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

    for train_index, eval_index in kf.split(data):
        train_data, eval_data = [data[i] for i in train_index], [data[i] for i in eval_index]
        train_labels, eval_labels = [labels[i] for i in train_index], [labels[i] for i in eval_index]

        # Train the classifier on the training data
        classifier.fit(train_data, train_labels)

        # Make predictions on the evaluation data
        predictions = classifier.predict(eval_data)

        # Calculate accuracy
        accuracy = calculate_accuracy(eval_labels, predictions)
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_accuracy


def calculate_accuracy(true_labels, predicted_labels):
    # Calculate the accuracy of predictions
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total = len(true_labels)
    accuracy = correct / total
    return accuracy


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
    # Task 5
    prediction = predict_sentiment("This movie was fantastic, exhilarating, and tremendously choreographed. " +
                                   "Close to perfection", prior_pos, prior_neg, likelihoods)
    # Task 6
    mean_accuracy = k_fold_cross_validation(classifier, training_data, training_labels, 10)


if __name__ == '__main__':
    main()
