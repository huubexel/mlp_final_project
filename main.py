from read_data import read_data_from_csv_file
from preprocessing import preprocesses_data, preprocessed_data_for_bert
from bert import get_bert_model, train


def main():
    bert_model_to_use = "bert-base-uncased"
    training_data = read_data_from_csv_file("data/train_data_offensive_taskA.csv")
    preprocessed_data = preprocesses_data(training_data)
    tokens_tensor, segments_tensor = preprocessed_data_for_bert(preprocessed_data, bert_model_to_use)
    bert_model = get_bert_model(bert_model_to_use)
    train(tokens_tensor, segments_tensor, bert_model)


if __name__ == "__main__":
    main()
