from read_and_write_data import read_data_from_csv_file, write_data_to_csv_file
from process_data import preprocesses_data, get_labels, get_mean_embeddings, get_as_numpy_array
from bert import get_bert_model, get_bert_tokenizer, get_embeddings_from_bert
from bagging import bagging


def main():
    """
    This performs a task of trying to guess whether a tweet is labeled as NOT or OFF,
    which is either not offensive or offensive, it does this by using machine learning.
    It gets data from the csv files, preprocesses that data, put the data into BERT.
    Then it gets the word embeddings from BERT and put these in the BaggingClassifier
    (after slightly adjusting them to fit). The BaggingClassifier predicts a label for
    each piece tweet in the test data set and writes this to a file.
    """

    # Preprocessing and setting up data for later usage
    bert_model_to_use = 'google-bert/bert-base-uncased'
    train_data_file = 'data/train_data_offensive_taskA.csv'
    test_data_file = 'data/test_data_text.csv'
    output_file = 'embeddings/predictions.csv'

    # Get the data that will be the input for BERT
    training_data = read_data_from_csv_file(train_data_file)
    preprocessed_train_data = preprocesses_data(training_data)
    train_labels = get_labels(training_data)

    test_data = read_data_from_csv_file(test_data_file)
    preprocessed_test_data = preprocesses_data(test_data)

    # Get the BERT model and tokenizer
    bert_model = get_bert_model(bert_model_to_use)
    bert_tokenizer = get_bert_tokenizer(bert_model_to_use)

    ###

    # Get the BERT embeddings
    train_embeddings = get_embeddings_from_bert(bert_model, bert_tokenizer, preprocessed_train_data)
    test_embeddings = get_embeddings_from_bert(bert_model, bert_tokenizer, preprocessed_test_data)

    # Convert the embeddings, so they can be used in the bagging classifier
    train_mean_embeddings = get_mean_embeddings(train_embeddings)
    test_mean_embeddings = get_mean_embeddings(test_embeddings)
    train_mean_embeddings_numpy = get_as_numpy_array(train_mean_embeddings)
    test_mean_embeddings_numpy = get_as_numpy_array(test_mean_embeddings)

    # Bagging ensemble
    results = bagging(train_mean_embeddings_numpy, train_labels, test_mean_embeddings_numpy)

    # Write the results to a csv file
    write_data_to_csv_file(output_file, results)


if __name__ == '__main__':
    main()
