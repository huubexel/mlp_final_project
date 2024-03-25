from torch import no_grad
from transformers import BertModel, BertTokenizer


def get_bert_model(bert_model_to_use: str):
    """ Returns the BERT model """
    return BertModel.from_pretrained(bert_model_to_use)


def get_bert_tokenizer(bert_model_to_use: str):
    """ Returns the BERT tokenizer """
    return BertTokenizer.from_pretrained(bert_model_to_use)


def get_embeddings_from_bert(model, tokenizer, preprocessed_data):
    """
        Tokenizes the data using the BERT model into BPE word-embeddings

        Args:
            model:              the BERT model
            tokenizer:          the BERT tokenizer
            preprocessed_data:  array of shape [N],
                N = the number of preprocessed tweets in the dataset

        Returns:
            outputs.last_hidden_state: tensor array of shape [N x F],
                N = number of data samples
                F = number of BERT embedding features for all words

    """

    # Tokenize the tweets so they fit in BERT
    tokenized_tweets = tokenizer(preprocessed_data,
                                 return_tensors='pt',
                                 truncation=True,
                                 padding=True,
                                 add_special_tokens=True)

    with no_grad():

        # Run the data through BERT
        outputs = model(**tokenized_tweets)

        return outputs.last_hidden_state
