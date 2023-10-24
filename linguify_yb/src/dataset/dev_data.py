"""Dataset Download Module

This module provides functions to download the Google dataset.

Functions:
- download_dataset(url: str, destination: str, path):
  Downloads a dataset from the given URL to the specified destination directory.
- main - the main function to run the script
"""

import os
import subprocess
import shutil
import zipfile
from loguru import logger  # from utils.logger import logger

# TODO Clean up code and complete doc
DATA_DIR = "data/raw/"
data_files = ["train.csv", "character_to_prediction_index.json"]
train_landmarks = ["1019715464.parquet", "1021040628.parquet", "105143404.parquet"]
TRAIN_LANDMARKS_DIR = "train_landmarks/"

COMMAND = [
    "kaggle",
    "competitions",
    "download",
    "-c",
    "asl-fingerspelling",
    "-f",
    "FILE",
    "-p",
    "data/raw/",
]


def check_storage(project=os.getcwd()):
    """_summary_

    Parameters
    ----------
    directory_path : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    StorageFullError
        _description_
    """
    total, used, free = shutil.disk_usage(project)
    total_size_gb = round(total / (2**30), 2)
    used_size_gb = round(used / (2**30), 2)
    free_size_gb = round(free / (2**30), 2)
    if used_size_gb / total_size_gb >= 0.8:
        raise StorageFullError
    else:
        return free_size_gb

class StorageFullError(Exception):
    """Custom exception for when storage is full."""

    pass
def downlaod_file(cmd, unzipped_file_path, data_dir):
    """_summary_

    Parameters
    ----------
    cmd : _type_
        _description_
    unzipped_file : _type_
        _description_
    data_dir : _type_
        _description_
    """
    subprocess.run(cmd, check=True, text=True)
    if (
        os.path.exists(unzipped_file_path)
        and os.path.splitext(unzipped_file_path)[1].lower() == ".zip"
    ):
        # Unzipping and delete the zipped file to free storage
        with zipfile.ZipFile(unzipped_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(unzipped_file_path)
    else:
        pass


def main():
    """_summary_"""
    logger.info("Commencing the data unzipping process")
    try:
        for file in data_files:
            logger.info(f"Downloading {file} in {DATA_DIR}")
            COMMAND[6] = file
            unzipfile_path = DATA_DIR + file + ".zip"
            downlaod_file(COMMAND, unzipfile_path, DATA_DIR)
            logger.info(f" {file} downloaded succesful")
        # c
        for parquet_file in train_landmarks:
            file_path = TRAIN_LANDMARKS_DIR + parquet_file
            COMMAND[6] = file_path
            COMMAND[8] = DATA_DIR + TRAIN_LANDMARKS_DIR
            unzipfile_path = DATA_DIR + file_path + ".zip"
            downlaod_file(COMMAND, unzipfile_path, DATA_DIR + TRAIN_LANDMARKS_DIR)
            logger.info(f"{parquet_file} downloaded succesful")

        logger.success("All files downloaded succesful")

    except Exception as error:
        logger.error(f"failed due to {error}")
        logger.exception("Data unloading was unsuccesfully")


if __name__ == "__main__":
    main()
