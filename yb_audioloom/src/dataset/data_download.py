"""doc
"""
from loguru import logger
import opendatasets as opd

DATA_URL = 'https://www.kaggle.com/datasets/rileydrizzy/a-multi-purpose-yoruba-speech-corpus/data'


def download_data():
    opd.download(DATA_URL, )

def main():
    logger.info("sujsfk")
    try:
        download_data()
        logger
    except Exception as error:


if __name__ == "__main__":
    main()
