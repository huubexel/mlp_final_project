from csv import reader


def read_data_from_csv_file(filename: str) -> list:
    """ Reads data from csv file and returns it """
    with open(filename, newline="", encoding="utf-8") as csv_file:      # Open .csv file, and put the data in csv_file
        data_rows = list(reader(csv_file))                              # read all the rows from the file in data_rows
    return data_rows
