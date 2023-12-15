"""
Module to define datasets and dataloaders for ASL Fingerspelling project.

Classes:
- TokenHashTable: A class for handling token-to-index and index-to-token mappings.
- LandmarkDataset: A dataset class for ASL Fingerspelling frames,\
    including methods for processing and cleaning frames.

Functions:
- read_file: Read data from file based on file_id_list and landmarks_metadata_path.
- get_dataset: Create a dataset with token-to-index mapping.
- prepare_dataloader: Prepare a dataloader with distributed sampling.
"""


import json
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from dataset.frames_config import FEATURE_COLUMNS
from dataset.preprocess import clean_frames_process

# File paths for metadata and phrase-to-index mapping
PHRASE_PATH = "/kaggle/input/asl-fingerspelling/character_to_prediction_index.json"
METADATA = "/kaggle/input/asl-fingerspelling/train.csv"

# Load phrase-to-index mapping
with open(PHRASE_PATH, "r", encoding="utf-8") as f:
    character_to_num = json.load(f)

# Define special tokens and their corresponding indices
PAD_TOKEN = "P"
START_TOKEN = "<"
END_TOKEN = ">"
PAD_TOKEN_IDX = 59
START_TOKEN_IDX = 60
END_TOKEN_IDX = 61

# Add special tokens to the mapping
character_to_num[PAD_TOKEN] = PAD_TOKEN_IDX
character_to_num[START_TOKEN] = START_TOKEN_IDX
character_to_num[END_TOKEN] = END_TOKEN_IDX

# Create a mapping from index to character
num_to_character = {j: i for i, j in character_to_num.items()}


class TokenHashTable:
    def __init__(
        self, word2index_mapping=character_to_num, index2word_mapping=num_to_character
    ):
        """
        Initialize a TokenHashTable to handle token-to-index and index-to-token mapping.

        Parameters:
            word2index_mapping (dict): Mapping from word to index.
            index2word_mapping (dict): Mapping from index to word.
        """
        self.word2index = word2index_mapping
        self.index2word = index2word_mapping

    def _indexesfromsentence(self, sentence):
        """
        Convert a sentence into a list of corresponding indices.

        Parameters:
            sentence (list): List of words in a sentence.

        Returns:
            list: List of indices corresponding to words in the sentence.
        """
        return [self.word2index[word] for word in sentence]

    def tensorfromsentence(self, sentence):
        """
        Convert a sentence into a tensor of indices.

        Parameters:
            sentence (list): List of words in a sentence.

        Returns:
            torch.Tensor: Tensor of indices.
        """
        indexes = self._indexesfromsentence(sentence)
        return torch.tensor(indexes, dtype=torch.long)

    def indexes_to_sentence(self, indexes_list):
        """
        Convert a list of indices into a list of corresponding words.

        Parameters:
            indexes_list (list or torch.Tensor): List or tensor of indices.

        Returns:
            list: List of words corresponding to the indices.
        """
        if torch.is_tensor(indexes_list):
            indexes_list = indexes_list.tolist()
        words = [self.index2word[idx] for idx in indexes_list]
        return words


def read_file(file_id_list, landmarks_metadata_path):
    """
    Read data from file based on file_id_list and landmarks_metadata_path.

    Parameters:
        file_id_list (list): List of tuples containing file paths and corresponding file_ids.
        landmarks_metadata_path (str): Path to the metadata file.

    Returns:
        tuple: A tuple containing lists of frames and phrases.
    """
    phrase_list = []
    frames_list = []
    for file, file_id in file_id_list:
        metadata_train_dataframe = pd.read_csv(landmarks_metadata_path)
        file_id_df = metadata_train_dataframe.loc[
            metadata_train_dataframe["file_id"] == file_id
        ]
        saved_parquet_df = pq.read_table(
            file, columns=["sequence_id"] + FEATURE_COLUMNS
        ).to_pandas()
        for seq_id, phrase in zip(file_id_df.sequence_id, file_id_df.phrase):
            frames = saved_parquet_df[saved_parquet_df.index == seq_id].to_numpy()
            # Handle NaN values
            frames_list.append(torch.tensor(frames))
            phrase_list.append(phrase)
    return frames_list, phrase_list


class LandmarkDataset(Dataset):
    def __init__(self, file_path, table, transform=True):
        """
        Initialize a LandmarkDataset.

        Parameters:
            - file_path (_type_): _description_
            - table (_type_): _description_
            - transform (bool, optional): _description_, by default True
        """
        self.landmarks_metadata_path = METADATA
        self.frames, self.labels = read_file(file_path, self.landmarks_metadata_path)
        self.trans = transform
        self.table = table

    def _label_pre(self, label_sample):
        """
        Preprocess label samples.

        Parameters:
            - label_sample (_type_): _description_

        Returns:
            - _type_: _description_
        """
        sample = START_TOKEN + label_sample + END_TOKEN
        new_phrase = self.table.tensorfromsentence(list(sample))
        ans = F.pad(
            input=new_phrase,
            pad=[0, 64 - new_phrase.shape[0]],
            mode="constant",
            value=PAD_TOKEN_IDX,
        )
        return ans

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        phrase = self.labels[idx]
        frames = self.frames[idx]

        if self.trans:
            phrase = self._label_pre(phrase)
            frames = clean_frames_process(frames)
        return frames, phrase


def get_dataset(file_path):
    """
    Create a dataset with token-to-index mapping.

    Parameters:
        - file_path (_type_): _description_

    Returns:
        - _type_: _description_
    """
    lookup_table = TokenHashTable(character_to_num, num_to_character)
    dataset = LandmarkDataset(file_path, lookup_table, transform=True)
    return dataset


def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers_: int = 1):
    """
    Prepare a dataloader with distributed sampling.

    Parameters:
        dataset (Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        num_workers_ (int, optional): Number of workers for data loading, by default 1.

    Returns:
        DataLoader: A DataLoader instance for the specified dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers_,
        sampler=DistributedSampler(dataset),
    )


#! A dataset class for debugging the train pipeline
class TestDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


#! Function to get a test dataset for debugging train pipeline
def get_test_dataset():
    dataset = TestDataset
    return dataset
