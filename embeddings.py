import random
import torch
from transformers import BertTokenizer, BertModel
import re
import csv

def preprocess_data(data):
    """
    Preprocesses the data for training by removing the punctuation, digits and user tags etc.

    Args:
        data:   array of shape [N x F],
                N = number of data samples (6818), this includes the row with column labels
                F = entries of the data (id, tweet text, annotation, annotator1, annotator2, annotator3, annotator4) (7)

    Returns:
        preprocessed_data:  array of shape [N],
                            N = Each preprocessed tweet text (6817)
    """
    # Removes the first row of the data set, which only included the column labels
    data = data[1:]

    preprocessed_data = []
    for tweet in data:
        tweet_text = tweet[1]
        # Removes hashtags, user tags and urls
        tweet_text = re.sub(r'(@|#|http)\S+', '', tweet_text)
        # Removes the emoji's from the tweet
        tweet_text = re.sub(r':[^\s]+:', '', tweet_text)
        # Removes punctuation
        tweet_text = re.sub(r'[^\w\s]', '', tweet_text)
        # Removes all digits
        tweet_text = re.sub(r'\d+', '', tweet_text)
        # Removes the string "url"
        tweet_text = tweet_text.replace("URL", "")
        # Makes all characters lowercase
        tweet_text = tweet_text.lower()
        # Removes the extra spaces at the beginning or end of the tweet text
        tweet_text = tweet_text.strip()

        preprocessed_data.append(tweet_text)

    return preprocessed_data


def bert_embed(preprocessed_data):
    # Set a random seed
    random_seed = 42
    random.seed(random_seed)

    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    print(f'this is the random seed: {random_seed}')


    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased') #.to('cuda')

    # Tokenize and encode text using batch_encode_plus
    # The function returns a dictionary containing the token IDs and attention masks
    encoding = tokenizer.batch_encode_plus(
    preprocessed_data,  # List of input texts
    padding = True,  # Pad to the maximum sequence length
    truncation = True,  # Truncate to the maximum sequence length if necessary
    return_tensors = 'pt',  # Return PyTorch tensors
    add_special_tokens = True  # Add special tokens CLS and SEP
    )

    input_ids = encoding['input_ids']  # Token IDs
    # print input IDs
    print(f"Input ID: {input_ids}")
    attention_mask = encoding['attention_mask']  # Attention mask
    # print attention mask
    print(f"Attention mask: {attention_mask}")

    # Generate embeddings using BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask) #.to('cuda')
        word_embeddings = outputs.last_hidden_state  # This contains the embeddings

    # Output the shape of word embeddings
    print(f"Shape of Word Embeddings: {word_embeddings.shape}")

    # # Decode the token IDs back to text
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # # print decoded text
    print(f"Decoded Text: {decoded_text}")

    # # Tokenize the text again for reference
    tokenized_text = tokenizer.tokenize(decoded_text)

    # # print tokenized text
    # print(f"tokenized Text: {tokenized_text}")

    # # Encode the text
    # encoded_text = tokenizer.encode(text, return_tensors='pt')  # Returns a tensor

    # # Print encoded text
    # print(f"Encoded Text: {encoded_text}")

    # Print word embeddings for each token
    for token, embedding in zip(tokenized_text, word_embeddings[0]):
        # print(f"Token: {token}")
        print(f"Embedding: {embedding}")
        print("\n")

    # Compute the average of word embeddings to get the sentence embedding
    sentence_embedding = word_embeddings.mean(dim=1)  # Average pooling along the sequence length dimension

    # Print the sentence embedding
    print("Sentence Embedding:")
    print(sentence_embedding)

    # Output the shape of the sentence embedding
    print(f"Shape of Sentence Embedding: {sentence_embedding.shape}")

    with open('train_bert_embeddings.txt', 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer_object = csv.writer(csv_file)

        # Write data to the CSV file
        csv_writer_object.writerows(sentence_embedding)









