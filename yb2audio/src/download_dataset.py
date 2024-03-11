"""Dataset Download Module

This module provides functions to download datasets from the IroyinSpeech dataset.

Functions:
- download_dataset(cmd: list):
    Downloads a dataset from Kaggle to the specified destination directory which is DATA_DIR.

- main():
    The main function to execute the dataset download script.

Usage:
    To download the dataset, run the script directly using Python.

Example:
    $ python src/download_dataset.py
"""

import subprocess
from utils.logging import logger

DATASET = "rileydrizzy/iroyinspeech"
DATA_DIR = "data/raw"

COMMAND = [
    "kaggle",
    "datasets",
    "download",
    f"{DATASET}",
    "-p",
    f"{DATA_DIR}",
    "--unzip",
]


def download_dataset(cmd):
    """
    Download the dataset from kaggle using it API.

    Parameters
    ----------
    cmd : list
        Kaggle API Commands.
    """
    subprocess.run(cmd, check=True, text=True)


def main():
    """
    main function to run the script
    """
    logger.info(f"Commencing downloading the dataset into {DATA_DIR}")
    try:
        download_dataset(cmd=COMMAND)
        logger.success(f"Dataset downloaded to {DATA_DIR} successfully.")
    except Exception as error:
        logger.exception(f"Dataset download failed due to: {error}")


if __name__ == "__main__":
    main()
