from read_and_write_data import read_data_from_csv_file, write_data_to_csv_file
from process_data import preprocesses_data, get_labels, get_mean_embeddings, get_as_numpy_array
from bert import get_bert_model, get_bert_tokenizer, get_embeddings_from_bert
from bagging import bagging


def main():
    """
    This program performs the task of classifying tweets as either being offensive (OFF) or not offensive (NOT).
    It does this by using the BERT word-embeddings obtained by running the bert-base-uncased model and extracting the
    word embeddings. The data is giving in a csv file with the following columns: id, text, offensive_aggregated,
    offense_a1,offense_a2,offense_a3,offense_a4. Of these columns we only use the text and offensive aggregated label.
    After obtaining the sentence embeddings, we train a bagging ensemble model on the training data. We fine-tune the
    model using the dev data and perform a classification on the test data.
    """

    # Specifying where to find the data we use and setting up for later usage
    bert_model_to_use = 'google-bert/bert-base-uncased'
    train_data_file = 'data/train_data_offensive_taskA.csv'
    test_data_file = 'data/test_data_text.csv'
    output_file = 'predictions.csv'

    # Reads the train and test data
    training_data = read_data_from_csv_file(train_data_file)
    test_data = read_data_from_csv_file(test_data_file)

    # Preprocesses the train and test data
    preprocessed_train_data = preprocesses_data(training_data)
    preprocessed_test_data = preprocesses_data(test_data)

    # Separates the labels from the other data in the training data set
    train_labels = get_labels(training_data)

    # Setting up the BERT model and tokenizer
    bert_model = get_bert_model(bert_model_to_use)
    bert_tokenizer = get_bert_tokenizer(bert_model_to_use)

    # Generates the BERT word-embeddings for both the train and test data
    train_embeddings = get_embeddings_from_bert(bert_model, bert_tokenizer, preprocessed_train_data)
    test_embeddings = get_embeddings_from_bert(bert_model, bert_tokenizer, preprocessed_test_data)

    # Converts the word-embeddings into sentence-embeddings by taking the mean of all word-embeddings in a sentence
    train_mean_embeddings = get_mean_embeddings(train_embeddings)
    test_mean_embeddings = get_mean_embeddings(test_embeddings)

    # Converts the embeddings from tensor arrays into numpy arrays
    train_mean_embeddings_numpy = get_as_numpy_array(train_mean_embeddings)
    test_mean_embeddings_numpy = get_as_numpy_array(test_mean_embeddings)

    # Training a bagging ensemble model on the training data set and using it
    # to obtain the predictions of the test data set
    predictions = bagging(train_mean_embeddings_numpy, train_labels, test_mean_embeddings_numpy)

    # Write the predictions to a csv file
    write_data_to_csv_file(output_file, predictions)


if __name__ == '__main__':
    main()
