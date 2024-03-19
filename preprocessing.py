import re
from torch import Tensor, tensor                # Tensor is the thing you are getting back from the function tensor()
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer


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
        tweet_text = tweet_text.replace("URL", "")                      # Removes all the URLs from the tweet
        tweet_text = tweet_text.lower()                                 # Makes the tweet lowercase
        tweet_text = tweet_text.strip()                                 # Strip excessive whitespaces from the tweet
        preprocessed_row.append(tweet_text)                             # Append the preprocessed tweet to the list

        preprocessed_row.extend(row[2:6])                               # Append the annotations to the list

        # Append the preprocessed row to the preprocessed data list
        preprocessed_data_list.append(preprocessed_row)

    return preprocessed_data_list


def preprocessed_data_for_bert(preprocessed_data: list, bert_model_to_use: str) -> (Tensor, Tensor):
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_to_use)

    indexed_tokens_list = []
    segment_ids_list = []

    longest_tweet_length = 0

    counter = 0
    for row in preprocessed_data:
        marked_tweet_text = "[CLS] " + row[1] + " [SEP]"

        # Tokenize the tweet in the BPE of BERT
        bpe_tokenized_tweet = bert_tokenizer.tokenize(marked_tweet_text)

        # Sentences can have at maximum 512 tokens, so any sentences that are longer than 512, we don't use.
        if len(bpe_tokenized_tweet) > 512:
            print("damn")

        if len(bpe_tokenized_tweet) > longest_tweet_length:
            longest_tweet_length = len(bpe_tokenized_tweet)
        if len(bpe_tokenized_tweet) > 512:
            print("damn")

        # Map the token strings to their vocabulary indeces. TODO find out why this is necessary.
        indexed_tokens_list.append(bert_tokenizer.convert_tokens_to_ids(bpe_tokenized_tweet))

        # TODO comment
        segment_ids_list.append([counter % 2] * len(bpe_tokenized_tweet))

        counter += 1

    # BERT needs the indexed_tokens (and therefore the segment_ids, because these need to be equal length) to be padded
    padded_indexed_tokens_list = pad_indexed_tokens_list(indexed_tokens_list, longest_tweet_length)
    padded_segment_ids_list = pad_segment_ids_list(segment_ids_list, longest_tweet_length)

    print(len(padded_indexed_tokens_list[0]))

    # Convert the lists to pytorch tensors because BERT needs these as inputs
    tokens_tensor = tensor(padded_indexed_tokens_list)
    segments_tensor = tensor(padded_segment_ids_list)

    return tokens_tensor, segments_tensor


def pad_indexed_tokens_list(indexed_tokens_list: list, longest_tweet_length: int) -> list:
    for indexed_token_list in indexed_tokens_list:
        indexed_token_list.extend([0] * (longest_tweet_length - len(indexed_token_list)))
    return indexed_tokens_list


def pad_segment_ids_list(segment_ids_list: list, longest_tweet_length: int) -> list:
    for segment_id_list in segment_ids_list:
        segment_id_list.extend([segment_id_list[0]] * (longest_tweet_length - len(segment_id_list)))
    return segment_ids_list

