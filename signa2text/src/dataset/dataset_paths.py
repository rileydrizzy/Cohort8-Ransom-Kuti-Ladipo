"""
Dataset Paths Module

This module provides functions to retrieve file paths for training and validation datasets.

Functions:
- get_dataset_paths(dev_mode=True): Retrieves paths for either development mode or the full dataset.

"""

import os
import json


def get_dataset_paths(dev_mode=True):
    """Get paths for training and validation datasets.

    Parameters
    ----------
    dev_mode : bool, optional
        If True, returns paths for development mode, else for full data.

    Returns
    -------
    list of tuple
        List of tuples containing file paths and corresponding file IDs for training
        and validation datasets.

    Raises
    ------
    AssertionError
        If the number of files retrieved does not match the expected count.
    """
    if dev_mode:
        dataset_paths = "data/dev_samples.json"
    else:
        dataset_paths = "data/dataset_paths.json"

    with open(dataset_paths, "r", encoding="utf-8") as json_file:
        dataset_paths_dict = json.load(json_file)

    # Training dataset
    train_dataset_dict = dataset_paths_dict["train_files"]
    train_file_ids = [os.path.basename(file_path) for file_path in train_dataset_dict]
    train_file_ids = [
        int(file_name.replace(".parquet", "")) for file_name in train_file_ids
    ]
    assert len(train_dataset_dict) == len(
        train_file_ids
    ), "Failed getting Train files path"
    train_ds_files = list(zip(train_dataset_dict, train_file_ids))

    # Validation dataset
    valid_dataset_dict = dataset_paths_dict["valid_files"]
    valid_file_ids = [os.path.basename(file_path) for file_path in valid_dataset_dict]
    valid_file_ids = [
        int(file_name.replace(".parquet", "")) for file_name in valid_file_ids
    ]
    assert len(valid_dataset_dict) == len(
        valid_file_ids
    ), "Failed getting Valid Files path"
    valid_ds_files = list(zip(valid_dataset_dict, valid_file_ids))

    return train_ds_files, valid_ds_files
