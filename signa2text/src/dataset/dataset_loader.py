"""
ASL Fingerspelling Dataset Module

This module defines classes and functions for handling datasets and dataloaders for the
ASL Fingerspelling project.

Classes:
- TokenHashTable: Handles token-to-index and index-to-token mappings.

- LandmarkDataset: Dataset class for ASL Fingerspelling frames.\
    Includes methods for processing and cleaning frames and phrase.

Functions:
- read_file(file_id_list, landmarks_metadata_path): Reads data from a file based on file IDs\
    and landmarks metadata path.

- get_dataset(file_path): Creates a dataset with a token-to-index mapping.

- prepare_dataloader(dataset, batch_size, num_workers_= 1): Prepares a dataloader
"""

import json
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from dataset.frames_config import FEATURE_COLUMNS
from dataset.preprocess import preprocess_frames

# File paths for metadata and phrase-to-index mapping
PHRASE_PATH = "kaggle/input/asl-fingerspelling/character_to_prediction_index.json"
METADATA = "kaggle/input/asl-fingerspelling/train.csv"

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
    """
    TokenHashTable handles token-to-index and index-to-token mappings for sequence data.

    This class is designed to facilitate the conversion between sequences of tokens
    and their corresponding indices, providing methods for transforming sentences
    to tensors and vice versa.
    """

    def __init__(
        self, word2index_mapping=character_to_num, index2word_mapping=num_to_character
    ):
        """Initialize a TokenHashTable.

        Parameters
        ----------
        word2index_mapping : dict, optional
            Mapping from word to index, by default character_to_num.
        index2word_mapping : dict, optional
            Mapping from index to word, by default num_to_character.
        """
        self.word2index = word2index_mapping
        self.index2word = index2word_mapping

    def _indexes_from_sentence(self, sentence):
        """Convert a sentence into a list of corresponding indices.

        Parameters
        ----------
        sentence : list
            List of words in a sentence.

        Returns
        -------
        list
            List of indices corresponding to words in the sentence.
        """
        return [self.word2index[word] for word in sentence]

    def tensor_from_sentence(self, sentence):
        """Convert a sentence into a tensor of indices.

        Parameters
        ----------
        sentence : list
            List of words in a sentence.

        Returns
        -------
        torch.Tensor
            Tensor of indices.
        """
        indexes = self._indexes_from_sentence(sentence)
        return torch.tensor(indexes, dtype=torch.long)

    def indexes_to_sentence(self, indexes_list):
        """Convert a list of indices into a list of corresponding words.

        Parameters
        ----------
        indexes_list : list or torch.Tensor
            List or tensor of indices.

        Returns
        -------
        list
            List of words corresponding to the indices.
        """
        if torch.is_tensor(indexes_list):
            indexes_list = indexes_list.tolist()
        words = [self.index2word[idx] for idx in indexes_list]
        return words


def read_file(file_id_list, landmarks_metadata_path):
    """
    Read data from files based on file IDs and landmarks metadata.

    Parameters
    ----------
    file_id_list : list
        List of tuples containing file paths and corresponding file IDs.
    landmarks_metadata_path : str
        Path to the metadata file.

    Returns
    -------
    tuple
        A tuple containing lists of frames and phrases.
    """
    phrase_list = []
    frames_list = []

    for file_path, file_id in file_id_list:
        metadata_train_dataframe = pd.read_csv(landmarks_metadata_path)
        file_id_df = metadata_train_dataframe.loc[
            metadata_train_dataframe["file_id"] == file_id
        ]

        saved_parquet_df = pq.read_table(
            file_path, columns=["sequence_id"] + FEATURE_COLUMNS
        ).to_pandas()

        for seq_id, phrase in zip(file_id_df.sequence_id, file_id_df.phrase):
            frames = saved_parquet_df[saved_parquet_df.index == seq_id].to_numpy()
            frames_list.append(torch.tensor(frames))
            phrase_list.append(phrase)

    return frames_list, phrase_list


class LandmarkDataset(Dataset):
    """
    LandmarkDataset represents a dataset of landmarks for sequence processing tasks.
    """

    def __init__(self, file_path, token_table, transform_=True):
        """
        Initialize a LandmarkDataset.

        Parameters
        ----------
        file_path : str or path-like
            Path to the dataset file.
        token_table : object
            An object representing a token table for phrase preprocessing.
        transform_ : bool, optional
            Indicates whether to apply transformations, by default True.
        """
        self.landmarks_metadata_path = METADATA
        self.frames, self.labels = read_file(file_path, self.landmarks_metadata_path)
        self.transform = transform_
        self.token_lookup_table = token_table

    def _phrase_preprocess(self, phrase_):
        """
        Tokenizes the input phrase.

        Parameters
        ----------
        phrase_ : str
            The original phrase

        Returns
        -------
        List[int]
            A list containing ints representing strings in the tokenized phrase.
        """
        phrase = START_TOKEN + phrase_ + END_TOKEN
        tokenize_phrase = self.token_lookup_table.tensor_from_sentence(list(phrase))
        tokenzie_phrase = F.pad(
            input=tokenize_phrase,
            pad=[0, 64 - tokenize_phrase.shape[0]],
            mode="constant",
            value=PAD_TOKEN_IDX,
        )
        return tokenzie_phrase

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a tuple containing frames and corresponding preprocessed phrase for a given index.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve from the dataset.

        Returns
        -------
        tuple
            A tuple containing frames and preprocessed labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        phrase = self.labels[idx]
        frames = self.frames[idx]

        if self.transform:
            phrase = self._phrase_preprocess(phrase)
            frames = preprocess_frames(frames)
        return frames, phrase


def get_dataset(file_path):
    """
    Create a dataset with token-to-index mapping.

    Parameters
    ----------
    file_path : str or path-like
        Path to the file containing the dataset.

    Returns
    -------
    dataset : LandmarkDataset
        An instance of LandmarkDataset with token-to-index mapping and frames.
    """

    lookup_table = TokenHashTable(character_to_num, num_to_character)
    dataset = LandmarkDataset(file_path, lookup_table, transform_=True)

    return dataset


def prepare_dataloader(dataset, batch_size, num_workers_=0):
    """
    Prepare a DataLoader

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to load.

    batch_size : int
        Number of samples per batch.

    num_workers_ : int, optional
        Number of workers for data loading, by default 0.

    Returns
    -------
    DataLoader
        A DataLoader instance for the specified dataset.

    Notes

    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers_,
    )
