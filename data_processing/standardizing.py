import pandas as pd
import numpy as np
import csv
import ast


def standardize(file_name, out_file):
    """
    Standardizes all data files by taking in file_name and subtracting all entries
    by the mean and dividing by the standard deviation.
    """
    data = pd.read_csv(file_name, header=None,
                       index_col=0, na_values=0,).reset_index()

    np.set_printoptions(suppress=True)
    all_data = np.zeros(13)
    for i in range(data.shape[0]):
        for n in range(data.shape[1]):
            row = np.array(ast.literal_eval(
                data[n][i].replace(" ", ",").replace("\n", "")))
            all_data = np.vstack((all_data, row))

    mu = np.mean(all_data[:, :12], axis=0)
    sigma = np.std(all_data[:, :12], axis=0)
    all_data[:, :12] -= mu
    all_data[:, :12] /= sigma
    all_data = all_data[1:]

    data_file = open(out_file, "w")
    writer_train = csv.writer(data_file)

    i = 0
    while i <= all_data.shape[0]:
        writer_train.writerow(all_data[i:i+24])
        i += 24
    data_file.close()


standardize("train_data.csv", "data_training_standardized")
standardize("val_data.csv", "data_validation_standardized")
standardize("test_data.csv", "data_testing_standardized")
