from read_data import read_data_from_csv_file
from preprocessing import preprocesses_data, preprocessed_data_for_bert, get_labels
from bert import get_bert_embeddings
from embeddings import write_to_file
from bagging import bagging


def main():
    # Preprocessing and setting up data for later usage
    bert_model_to_use = "bert-base-uncased"
    training_data = read_data_from_csv_file("data/train_data_offensive_taskA.csv")
    preprocessed_data = preprocesses_data(training_data)
    labels = get_labels(training_data)

    # BERT
    tokens_tensor, segments_tensor = preprocessed_data_for_bert(preprocessed_data, bert_model_to_use)
    embeddings = get_bert_embeddings(bert_model_to_use, tokens_tensor, segments_tensor)
    write_to_file(embeddings)

    # Ensemble model
    bagging(embeddings, labels)


if __name__ == "__main__":
    main()
