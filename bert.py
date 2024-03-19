from torch import no_grad, Tensor, device
from torch.cuda import is_available
from transformers import BertModel


def get_bert_model(bert_model_to_use: str):
    """ Get the BERT model """

    # If you have a GPU that has cuda and has that enabled run it on there, otherwise use the CPU
    device_to_train_on = device('cuda' if is_available() else 'cpu')

    # Get the pretrained model
    model = BertModel.from_pretrained(bert_model_to_use, output_hidden_states=True).to(device_to_train_on)

    # This sets the training mode on false, so BERT won't retrain itself, which is what we want
    model.eval()

    return model


def train(tokens_tensor: Tensor, segments_tensor: Tensor, model):
    with no_grad():
        # Train the model
        outputs = model(tokens_tensor, segments_tensor)

        hidden_states = outputs[2]

        print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        layer_i = 0

        print("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0

        print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0

        print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
