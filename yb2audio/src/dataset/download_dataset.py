"""Dataset Download Module

This module provides functions to download datasets from a IroyinSpeech dataset.

Functions:
- download_dataset(url: str, destination: str) -> bool:
  Downloads a dataset from the given URL to the specified destination directory.
- main - the main function to run the script
"""

import os

import opendatasets as opd
from loguru import logger

# TODO refactor with config.yaml
URL_ = "https://www.kaggle.com/datasets/rileydrizzy/iroyinspeech"
DATA_DIR = "data/raw"


def download_dataset_(url, destination_dir):
    """download the dataset from kaggle api

    Parameters
    ----------
    url : str
        dataset kaggle url
    destination_dir : str, path
        directory to download the dataset into
    """

    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir)
    opd.download_kaggle_dataset(url, destination_dir)


def main():
    """main function to run the script"""
    logger.info(f"Commencing downloading the dataset into {DATA_DIR}")
    try:
        download_dataset_(url=URL_, destination_dir=DATA_DIR)
        logger.succes(f"Dataset downloaded to {DATA_DIR} successfully.")
    except Exception as error:
        logger.error(f"Dataset download failed due to: {error}")


if __name__ == "__main__":
    main()
