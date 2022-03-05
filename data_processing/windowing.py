import pandas as pd
import numpy as np
import ast
import csv
import sklearn.model_selection

# read from labelled_monthly.csv and set print rules
np.set_printoptions(threshold=np.inf)
pd.options.display.max_colwidth = 999999999
pd.options.display.max_columns = 999999999
pd.options.display.width = 999999999
pd.options.display.max_rows = 99999999

data = pd.read_csv("labelled_monthly.csv", header=None,
                   index_col=0, na_values=0).reset_index()
data = data.replace(np.nan, 0)


def data_subset(career: np.array, player_id):
    """
    Create subsets of an entire player's career for
    players with 4 seasons + worth of experience.
    """
    windows = []
    months = career.shape[0]
    i = 0
    while (i+24) <= months:
        windows.append(career[i:i+24])
        i += 6
    return windows


def split_train_val_test(data):
    """
    Split the data into respective training (60%), validation (20%)
    and test (20%) sets. 

    """
    ID_trainval, ID_test = sklearn.model_selection.train_test_split(
        data, test_size=0.2, train_size=0.8)
    ID_train, ID_val = sklearn.model_selection.train_test_split(
        ID_trainval, test_size=0.25, train_size=0.75)

    return ID_train, ID_val, ID_test


def write_window_stats(file_name):
    """
    Writes all the windowed statistics to the file named file_name.
    """
    file = open(file_name, "w")
    writer_file = csv.writer(file)
    for ids in ID_train:
        for subsets in data_subsets[ids]:
            writer_file.writerow(subsets)
    file.close()


# format the subsets for each player's full careers
data_subsets = {}
data_no_subsets = {}
for player in range(data.shape[0]):
    np.set_printoptions(threshold=np.inf)
    s = " ".join(data[1][player].split()).replace(" ", ",").replace(
        "[,", "[").replace(",]", "]").replace("nan", "0")
    full_career = np.array(ast.literal_eval(s), dtype=object)
    if full_career.shape[0] >= 24:
        data_no_subsets[data[0][player]] = [full_career]
        if full_career.shape[0] > 24:
            data_subsets[data[0][player]] = data_subset(full_career, player)

# split the data by player_id to avoid cross-contamination
ID_train, ID_val, ID_test = split_train_val_test(list(data_subsets.keys()))

# write to train_data.csv, val_data.csv and test_data.csv
write_window_stats("train_data.csv")
write_window_stats("val_data.csv")
write_window_stats("test_data.csv")
