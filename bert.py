from torch import no_grad
from transformers import BertModel, BertTokenizer


def get_bert_model(bert_model_to_use: str):
    """ Returns the BERT model """
    return BertModel.from_pretrained(bert_model_to_use)


def get_bert_tokenizer(bert_model_to_use: str):
    """ Returns the BERT tokenizer """
    return BertTokenizer.from_pretrained(bert_model_to_use)


def get_embeddings_from_bert(model, tokenizer, preprocessed_data):
    """ Tokenize the tweets so BERT can read them and run the model, return the outputs (embeddings) of the model """

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
