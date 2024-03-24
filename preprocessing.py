from re import sub
from transformers import AutoTokenizer


def preprocesses_data(data: list) -> (list, list):

    # Remove the first line in the dataset because it is metadata
    data = data[1:]

    labels = []
    preprocessed_data = []
    for tweet in data:

        # Get the label to return later on
        labels.append(tweet[2])

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

        tweet_text = tweet_text.replace("_", '')

        # Makes all characters lowercase
        tweet_text = tweet_text.lower()

        # Removes the extra spaces at the beginning or end of the tweet text
        tweet_text = tweet_text.strip()

        preprocessed_data.append('[CLS] ' + tweet_text + ' [SEP]')

    return preprocessed_data, labels


def preprocessed_data_for_bert(preprocessed_data: list, bert_model_to_use: str, device_to_run_on):

    # Get the tokenizer from BERT that can do the BERT BPE and BERT indexing
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_to_use)

    # Tokenize the tweet in the BPE of BERT
    bpe_tokenized_tweet = (bert_tokenizer(preprocessed_data,
                                          return_tensors='pt',
                                          truncation=True,
                                          padding=True,
                                          add_special_tokens=True).to(device_to_run_on))

    return bpe_tokenized_tweet
