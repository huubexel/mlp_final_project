from re import sub
from torch import Tensor, tensor                # Tensor is the object you are getting back from the function tensor()
from transformers import BertTokenizer


def get_labels(data: list):
    return [row[2] for row in data[1:]]


def preprocesses_data(data: list) -> list:
    """ TODO maybe change this text: This preprocesses the tweets in the list, the structure of the list stays the same """

    # Remove the first line in the dataset because it is metadata
    data = data[1:]

    preprocessed_data = []
    for tweet in data:
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

        preprocessed_data.append(tweet_text)

    return preprocessed_data


def remove_punctuation(tweet_text: list[str]) -> list[str]:
    """ This removes the punctuation in the tweet for every token in the tweet """
    return [sub(r'[^\w\s]', '', part_of_tweet) for part_of_tweet in tweet_text]


def preprocessed_data_for_bert(preprocessed_data: list, bert_model_to_use: str) -> (Tensor, Tensor):
    """
    If you ask yourself, why does BERT need certain things, hey I didn't make BERT either, so I cannot
    answer that question. But I am here to answer the question: what does BERT need?
    - First of all BERT needs all the sentences to start with [CLS] and end with [SEP]
    - Than BERT has a special vocabulary, these are the BPE words, that is the language that BERT speaks
      an example of that is for example the word 'embeddings' will look like this: ['em', '##bed', '##ding', '##s']
    - The bert tokenizer tokenize function needs whole sentences, the [CLS] and [SEP] already need to be in the
      sentence when the tokenize function is used, that is why the [CLS] and [SEP] are appended before using tokenize()
    - This bert tokenizer tokenize needs whole sentences, that is why sent_tokenize is used from NLTK
    - BERT cannot handle sentences that have more than 512 BPE tokens, so sentences that are longer are not used.
    - You cannot give BERT lists or dictionaries, it won't accept these. You have to give it Tensors. These tensors
      are just another way of representing your data, they are still lists but a bit more fancy and probably with
      a bit more functionality. PyTorch is the library that we use to make these Tensors, with the tensor function.
    - You know how I specified that BERT has BPE as vocabulary, well either BERT or PyTorch cannot handle those
      directly, that's why you have to convert them into indexed tokens, these are the id's for the BPE words where
      each word has a certain id in the vocabulary. convert_tokens_to_ids() takes care of this on sentence level.
    - I really have no idea why BERT and/or PyTorch needs this, but BERT and/or PyTorch needs to have an alternating
      list of either all 0's or 1's per sentence where it is like: sentence1 -> 0's sentence2 -> 1's sentence3 -> 0's
      sentence4 -> 1's sentence5 -> 0's etc.
    - Either PyTorch or BERT needs to have all dimensions in the indexed_tokens_list and segment_ids_list padded,
      so we pad each list until they have reached the longest_tweet_length in these two functions:
      pad_indexed_tokens_list, pad_segment_ids_list. If you have done all of the above, the tensor() function
      will accept your input and give you Tensor objects which you can feed to BERT.

    This function takes the normally preprocessed data and returns two Tensor objects which are created from
    this normally preprocessed data. You can feed this to BERT!
    """

    # Get the tokenizer from BERT that can do the BERT BPE and BERT indexing
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_to_use)

    # lists where the indexed tokens and segments ids will go
    indexed_tokens_list = []
    segments_ids_list = []

    # After all rows, this will hold the longest_tweet_length that is under 512 BPE words length
    longest_tweet_length = 0

    # Used to set either 0's or 1's in the segment_ids list for each row
    counter = 0
    new_batch_size = 8

    for row in preprocessed_data:
        # BERT needs a [CLS] at the start of the sentence and [SEP] at the end of the sentence in order to be able
        # to read the sentence, so we append that here
        marked_tweet_text = '[CLS] ' + row + ' [SEP]'

        # Tokenize the tweet in the BPE of BERT
        bpe_tokenized_tweet = bert_tokenizer(marked_tweet_text, return_tensors='pt', truncation=True, padding=True)

        # Using a small batch size makes it so that weak processors can process this code without getting an error
        bpe_tokenized_tweet['input_ids'] = bpe_tokenized_tweet['input_ids'].repeat(new_batch_size, 1)
        bpe_tokenized_tweet['attention_mask'] = bpe_tokenized_tweet['attention_mask'].repeat(new_batch_size, 1)

        # Sentences can have max 512 BPE tokens, so any sentences that are longer than 512, we cut behind 512
        if len(bpe_tokenized_tweet) > 512:
            bpe_tokenized_tweet = bpe_tokenized_tweet[:511] + ['[SEP]']

        # If the current tweet is longer than longest_tweet_length, this tweets length becomes the longest_tweet_length
        if len(bpe_tokenized_tweet) > longest_tweet_length:
            longest_tweet_length = len(bpe_tokenized_tweet)

        # Map the BPE words to their corresponding id's in the vocabulary of BERT and put them in indexed_tokens_list
        indexed_tokens_list.append(bert_tokenizer.convert_tokens_to_ids(bpe_tokenized_tweet))

        # Make the segment ids list for this sentence and append it to the segment_ids_list
        segments_ids_list.append([counter % 2] * len(bpe_tokenized_tweet))

        counter += 1

    # Convert the lists to PyTorch Tensors because BERT needs these as inputs
    tokens_tensor = tensor(indexed_tokens_list)
    segments_tensor = tensor(segments_ids_list)

    return tokens_tensor, segments_tensor
