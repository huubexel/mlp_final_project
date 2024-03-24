from torch import no_grad, Tensor, device
from torch.cuda import is_available, empty_cache
from transformers import BertModel


def get_bert_embeddings(bert_model_to_use: str, tokens_tensor: Tensor, segments_tensor: Tensor):
    """ Get the BERT model """

    # If you have a GPU that has cuda and has that enabled run it on there, otherwise use the CPU
    device_to_train_on = device('cuda' if is_available() else 'cpu')

    empty_cache()

    # Get the pretrained model
    model = BertModel.from_pretrained(bert_model_to_use, output_hidden_states=True).to(device_to_train_on)

    # This sets the training mode on false, so BERT won't retrain itself, which is what we want
    model.eval()

    with no_grad():
        # Run the data through BERT
        outputs = model(tokens_tensor.to(device_to_train_on), segments_tensor.to(device_to_train_on))

        return outputs[2][11]
