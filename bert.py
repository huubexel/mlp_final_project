from torch import no_grad
from transformers import BertModel


def get_bert_model(bert_model_to_use: str, device_to_run_on):
    # Get the pretrained model
    model = BertModel.from_pretrained(bert_model_to_use).to(device_to_run_on)

    # This sets the training mode on false, so BERT won't retrain itself, which is what we want
    model.eval()

    return model


def get_bert_embeddings(model, tokenized_tweets, device_to_run_on):
    with no_grad():

        # Run the data through BERT
        outputs = model(**tokenized_tweets).to(device_to_run_on)

        return outputs.last_hidden_state
