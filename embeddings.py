import csv
from torch import mean


def write_to_file(embeddings):

    tensor_list = []
    for embeddings_tensor in embeddings:
        tensor_list.append(mean(embeddings_tensor, dim=0).tolist())

    with open('embeddings.txt', 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer_object = csv.writer(csv_file)

        # Write data to the CSV file
        csv_writer_object.writerows(tensor_list)

    return
