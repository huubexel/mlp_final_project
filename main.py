from read_data import read_data_from_csv_file
from preprocessing import preprocesses_data, preprocessed_data_for_bert
from bert import get_bert_model, get_bert_embeddings
from embeddings import write_to_file
from torch import device
from torch.cuda import is_available
from bagging import bagging


def main():
    # Get the device to run on which is either the cpu or the cuda cores from the gpu
    # device_to_run_on = device('cuda' if is_available() else 'cpu')
    device_to_run_on = 'cpu'

    # Preprocessing and setting up data for later usage
    bert_model_to_use = "google-bert/bert-base-uncased"
    training_data = read_data_from_csv_file("data/train_data_offensive_taskA.csv")
    preprocessed_data, labels = preprocesses_data(training_data)

    tokenized_tweets = preprocessed_data_for_bert(preprocessed_data, bert_model_to_use, device_to_run_on)
    bert_model = get_bert_model(bert_model_to_use, device_to_run_on)

    embeddings = get_bert_embeddings(bert_model, tokenized_tweets, device_to_run_on)
    write_to_file(embeddings)

    # Ensemble model
    # bagging(embeddings, labels)


if __name__ == "__main__":
    main()
