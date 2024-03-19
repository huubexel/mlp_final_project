import re
from nltk.tokenize import sent_tokenize


def preprocesses_data(data: list) -> list:
    """ This preprocesses the tweets in the list, the structure of the list stays the same """
    preprocessed_data_list = []

    for row in data:
        # Make a list where the row number, the preprocessed tweet and the annotation go in.
        preprocessed_row = [row[0]]

        # Preprocess the tweet
        tweet_text = row[1]                                             # Put the tweet text in a var for readability
        tweet_text = re.sub(r"@[\w_]+", "", tweet_text)                 # Removes all user tags (@USER) from the tweet
        tweet_text = re.sub(r"\d+", "", tweet_text)                     # Removes all numbers from the tweet
        tweet_text = re.sub(r"#\w+", "", tweet_text)                    # Removes all hashtags from the tweet
        tweet_text = sent_tokenize(tweet_text, language="dutch")[0]     # Removes all punctuation from the tweet
        tweet_text = tweet_text.lower()                                 # Makes the tweet lowercase
        tweet_text = tweet_text.strip()                                 # Strip excessive whitespaces from the tweet
        preprocessed_row.append(tweet_text)                             # Append the preprocessed tweet to the list

        preprocessed_row.extend(row[2:6])                               # Append the annotations to the list

        # Append the preprocessed row to the preprocessed data list
        preprocessed_data_list.append(preprocessed_row)

    return preprocessed_data_list
