import os

import pandas as pd


def get_train_val_test_paths(dataset_folder, train_filename="train_split.csv", val_filename="val_split.csv",
                             test_filename="test_split.csv"):
    train_path = os.path.join(dataset_folder, train_filename)
    val_path = os.path.join(dataset_folder, val_filename)
    test_path = os.path.join(dataset_folder, test_filename)
    return train_path, val_path, test_path


def calculate_max_protein_length(dataset_folder):
    paths = get_train_val_test_paths(dataset_folder)
    max_len = -1
    for path in paths:
        curr_max = pd.read_csv(path)["protein_length"].max()
        if curr_max > max_len:
            max_len = curr_max
    print("Max protein length:", max_len)
    return max_len
