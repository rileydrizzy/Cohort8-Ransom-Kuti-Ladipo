"""Dataset Download Module

This module provides functionality to download a subsample of the Google ASL dataset through
the Kaggle API. It includes functions to download specific files, check available storage space, 
and a main script to orchestrate the downloading process.

Functions:
- download_dataset(url: str, destination: str, path: str):
  Downloads a dataset from the given URL to the specified destination directory.

- check_storage(project_dir: str = os.getcwd()) -> float:
  Checks and returns the available storage space in gigabytes (GB) for a specified directory.

- download_file(cmd: List[str], unzipped_file_path: str, data_dir: str):
  Downloads a file using Kaggle API commands and unzips it.

- main():
  The main function to execute the script. It orchestrates the download of various dataset files\
    and logs progress.

Constants:
- DATA_DIR: Default directory for storing downloaded data.
- DATA_FILES: List of files to download from the Kaggle competition.
- train_landmarks: List of additional files related to landmarks for training data.
- TRAIN_LANDMARKS_DIR: Directory for storing downloaded landmark files.
- COMMAND: Kaggle API command template used for downloading files.

Custom Exception:
- StorageFullError: Raised when the available storage space is insufficient.

Usage:
- Ensure the Kaggle API is configured.
- Execute the script to download the specified dataset files.

Example:
    $ python src/download_dev_data.py

"""


import os
import shutil
import subprocess
import zipfile
from utils.logging import logger

DATA_DIR = "kaggle/input/asl-fingerspelling/"
DATA_FILES = ["train.csv", "character_to_prediction_index.json"]
train_landmarks = ["1019715464.parquet", "1021040628.parquet", "105143404.parquet"]
TRAIN_LANDMARKS_DIR = "train_landmarks/"

COMMAND = [
    "kaggle",
    "competitions",
    "download",
    "-c",
    "asl-fingerspelling",
    "-f",
    "FILE_NAME",
    "-p",
    f"{DATA_DIR}",
]


class StorageFullError(Exception):
    """Custom exception for when storage is full."""

    pass


def check_storage(project_dir=os.getcwd()):
    """Check and return available storage space.

    Parameters
    ----------
    project_dir : str, Path
        Current working directory or directory path.

    Returns
    -------
    int
        The size of available storage space (GB).

    Raises
    ------
    StorageFullError
        Exception for when storage is full.
    """
    total, used, free = shutil.disk_usage(project_dir)
    total_size_gb = round(total / (2**30), 2)
    used_size_gb = round(used / (2**30), 2)
    free_size_gb = round(free / (2**30), 2)

    if used_size_gb / total_size_gb >= 0.8:
        raise StorageFullError("Storage is full. Cannot perform the operation.")
    return free_size_gb


def download_file(cmd, zipped_file_path, data_dir):
    """Download file using Kaggle API.

    Parameters
    ----------
    cmd : list
        Kaggle API Commands.
    zipped_file_path : str, Path
        Path of the unzipped file.
    data_dir : str, Path
        The directory where the data should be downloaded into.
    """
    subprocess.run(cmd, check=True, text=True)
    if (
        os.path.exists(zipped_file_path)
        and os.path.splitext(zipped_file_path)[1].lower() == ".zip"
    ):
        # Unzipping and delete the zipped file to free storage
        with zipfile.ZipFile(zipped_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zipped_file_path)


def main():
    """
    Orchestrates the dataset download using the Kaggle API.

    Downloads specified dataset and landmark files, logging progress and checking storage space.

    Raises
    ------
    Exception
        If an error occurs during the download process.
    """
    logger.info("Commencing downloading the dataset")
    try:
        logger.info(f"Current Available space {check_storage()}GB")

        # Downloading the metadata files
        for file in DATA_FILES:
            logger.info(f"Downloading {file} in {DATA_DIR}")

            # Swtiching "FILE_NAME" in cmd list with the actual file name in kaggle
            COMMAND[6] = file
            zip_file_path = DATA_DIR + file + ".zip"
            download_file(COMMAND, zip_file_path, DATA_DIR)
            logger.info(f"{file} downloaded successfully")

        # Swtiching the directory to download the landmarks into
        COMMAND[8] = DATA_DIR + TRAIN_LANDMARKS_DIR

        # Downloading the LANDMARKS files
        for parquet_file in train_landmarks:
            logger.info(f"Current Available space {check_storage()}GB")
            file_path = TRAIN_LANDMARKS_DIR + parquet_file

            # Swtiching "FILE_NAME" in cmd list with the actual file name in kaggle
            COMMAND[6] = file_path
            zip_file_path = DATA_DIR + file_path + ".zip"
            download_file(COMMAND, zip_file_path, DATA_DIR + TRAIN_LANDMARKS_DIR)
            logger.info(f"{parquet_file} downloaded successfully")

        logger.success("All files downloaded successfully")

    except Exception as error:
        logger.exception(f"Data unloading was unsuccessful due to {error}")


if __name__ == "__main__":
    main()
