from re import sub

import numpy as np
from torch import mean


def preprocesses_data(data: list) -> (list, list):
    """ This processes the way in a way"""

    # Remove the first line in the dataset because it is metadata
    data = data[1:]

    preprocessed_data = []
    for tweet in data:
        # Get the label to return later on

        # Put the tweet text in a variable
        tweet_text = tweet[1]

        # Removes hashtags, user tags and urls
        tweet_text = sub(r'(@|#|http)\S+', '', tweet_text)

        # Removes the emoji's from the tweet
        tweet_text = sub(r':[^\s]+:', '', tweet_text)

        # Removes punctuation
        tweet_text = sub(r'[^\w\s]', '', tweet_text)

        # Removes all digits
        tweet_text = sub(r'\d+', '', tweet_text)

        # Removes the string "url"
        tweet_text = tweet_text.replace('URL', '')

        tweet_text = tweet_text.replace('_', '')

        # Makes all characters lowercase
        tweet_text = tweet_text.lower()

        # Removes the extra spaces at the beginning or end of the tweet text
        tweet_text = tweet_text.strip()

        preprocessed_data.append(tweet_text)

    return preprocessed_data


def get_labels(data: list) -> list:
    """ Returns all the labels in the data (excluding the first metadata line) """
    return [tweet[2] for tweet in data[1:]]


def get_mean_embeddings(embeddings):
    """ For each tweet, get the mean of all the word embeddings, append this to the list, return the list """
    return [mean(embeddings_tensor, dim=0).tolist() for embeddings_tensor in embeddings]


def get_as_numpy_array(embeddings):
    """ Makes a numpy array out of the embeddings and returns that """
    return np.array(embeddings)
