import numpy as np
from re import sub
from torch import mean


def preprocesses_data(data: list) -> list:
    """
        Preprocesses the data by extracting the tweets and removing things like hashtags, urls, user tags,
        emoji's and punctuation from the text.

        Args:
            data:   array of shape [N x F],
                N = number of data samples, this includes the row with column labels
                F = entries of the data

        Returns:
            data_rows:  array of shape [N],
                N = number of preprocessed tweets
    """

    # Remove the first line in the dataset because it includes the column labels
    data = data[1:]

    preprocessed_data = []
    for tweet in data:

        # Extracts the tweet texts from the data
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

        # Removes all occurrences of underscore in the text
        tweet_text = tweet_text.replace('_', '')

        # Makes all characters lowercase
        tweet_text = tweet_text.lower()

        # Removes the extra spaces at the beginning or end of the tweet text
        tweet_text = tweet_text.strip()

        preprocessed_data.append(tweet_text)

    return preprocessed_data


def get_labels(data: list) -> list:
    """
        Extracts the labels from the data set, these are used to train the classifier

        Args:
            data:   array of shape [N x F],
                N = number of data samples, this includes the row with column labels
                F = entries of the data

        Returns:
            data_rows:  array of shape [N],
                N = number of extracted labels
    """
    return [tweet[2] for tweet in data[1:]]


def get_mean_embeddings(embeddings):
    """
        Extracts the labels from the data set, these are used to train the classifier

        Args:
            embeddings:   tensor array of shape [N x F],
                N = number of data samples
                F = number of BERT embedding features for all words

        Returns:
            list:  tensor array of shape [N x F],
                N = number of data samples
                F = number of BERT embedding features for each sentence
    """
    return [mean(embeddings_tensor, dim=0).tolist() for embeddings_tensor in embeddings]


def get_as_numpy_array(embeddings):
    """
        Converts the embeddings into a numpy array

        Args:
            embeddings: tensor array

        Returns:
            np.array(embeddings):  numpy array
    """
    return np.array(embeddings)
