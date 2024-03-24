from embeddings import preprocess_data, bert_embed
from csv import reader


def read_data_from_csv_file(filename):
    """
        Reads data from csv file and returns it

        Args:
            filename:   string of csv file name

        Returns:
            data_rows:  array of shape [N x F],
                        N = number of data samples (6818), this includes the row with column labels
                        F = entries of the data (id, tweet text, annotation, annotator1, annotator2, annotator3, annotator4) (7)
    """
    with open(filename, newline="", encoding="utf-8") as csv_file:
        data_rows = list(reader(csv_file))
    return data_rows


def main():
    training_data = read_data_from_csv_file("data/train_data_offensive_taskA.csv")

    # Preprocesses the tweet texts
    preprocessed_data = preprocess_data(training_data)

    # Gets the bert embeddings
    bert_embed(preprocessed_data)



if __name__ == "__main__":
    main()
