"""doc

functions:
    *unzip_file - unzip the zip file and export it
    *main - the main function to run the script
"""

from loguru import logger
import opendatasets as opd

DATA_URL = 'https://www.kaggle.com/datasets/rileydrizzy/a-multi-purpose-yoruba-speech-corpus/data'


def download_data(DATA_DIR):
    opd.download(DATA_URL, DATA_DIR)

def main():
    logger.info("Commencing the data unzipping process")
    try:
        download_data()
        logger
    except Exception as error:
        pass

if __name__ == "__main__":
    main()
