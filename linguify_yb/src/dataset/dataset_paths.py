"""doc
"""
import os
import json
from utils.logger_util import logger


def get_dataset_paths():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    try:
        # On kaggle replace with "data/dataset_paths.json" to train on full data
        dataset_paths = "data/dev_samples.json"
        with open(dataset_paths, "r", encoding="utf-8") as json_file:
            dataset_paths_dict = json.load(json_file)

        # Training dataset
        train_dataset_dict = dataset_paths_dict["train_files"]
        train_file_ids = [os.path.basename(file) for file in train_dataset_dict]
        train_file_ids = [
            int(file_name.replace(".parquet", "")) for file_name in train_file_ids
        ]
        assert len(train_dataset_dict) == len(
            train_file_ids
        ), "Failed getting Train files path"
        train_ds_files = list(zip(train_dataset_dict, train_file_ids))

        # Validation dataset
        valid_dataset_dict = dataset_paths_dict["valid_files"]
        valid_file_ids = [os.path.basename(file) for file in valid_dataset_dict]
        valid_file_ids = [
            int(file_name.replace(".parquet", "")) for file_name in valid_file_ids
        ]
        assert len(train_dataset_dict) == len(
            train_file_ids
        ), "Failed getting of Valid Files path"
        valid_ds_files = list(zip(valid_dataset_dict, valid_file_ids))

        return train_ds_files, valid_ds_files
    except AssertionError as asset_error:
        logger.exception(f"Failed due to {asset_error}")
