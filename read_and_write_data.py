import csv


def read_data_from_csv_file(filename: str) -> list:
    """ Reads data from csv file and returns it """
    with open(filename, newline='', encoding='utf-8') as csv_file:      # Open .csv file, and put the data in csv_file
        data_rows = list(csv.reader(csv_file))                          # read all the rows from the file in data_rows
    return data_rows                                                    # Return the data just


def write_data_to_csv_file(filename: str, data_to_write) -> None:
    """ Writes the given data to the csv file as: id, offensive_aggregated """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)                                    # Get the csv writer
        writer.writerow(['id', 'offensive_aggregated'])                 # Write the two headers
        for index in range(0, len(data_to_write)):
            writer.writerow([index, data_to_write[index]])
