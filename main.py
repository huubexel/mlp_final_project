from read_data import read_data_from_csv_file
from preprocessing import preprocesses_data


def main():
    training_data = read_data_from_csv_file("data/train_data_offensive_taskA.csv")
    preprocessed_data = preprocesses_data(training_data)


if __name__ == "__main__":
    main()
