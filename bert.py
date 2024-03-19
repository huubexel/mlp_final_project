from torch import no_grad, Tensor
from transformers import BertModel


def get_bert_model(bert_model_to_use: str):
    model = BertModel.from_pretrained(bert_model_to_use, output_hidden_states=True)
    model.eval()
    return model


def train(tokens_tensor: Tensor, segments_tensor: Tensor, model):
    print(tokens_tensor)
    # with no_grad():
    #     outputs = model(tokens_tensor, segments_tensor)
    #
    #     hidden_states = outputs[2]
    #
    #     print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    #     layer_i = 0
    #
    #     print("Number of batches:", len(hidden_states[layer_i]))
    #     batch_i = 0
    #
    #     print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    #     token_i = 0
    #
    #     print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

