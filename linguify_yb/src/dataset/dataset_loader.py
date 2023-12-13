"""doc
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.dataset.frames_config import FEATURE_COLUMNS, FRAME_LEN
from src.dataset.preprocess import clean_frames_process

PHRASE_PATH = "kaggle/input/asl-fingerspelling/character_to_prediction_index.json"
METADATA = "/kaggle/input/asl-fingerspelling/train.csv"

with open(PHRASE_PATH, "r", encoding="utf-8") as f:
    character_to_num = json.load(f)

PAD_TOKEN = "P"
START_TOKEN = "<"
END_TOKEN = ">"
PAD_TOKEN_IDX = 59
START_TOKEN_IDX = 60
END_TOKEN_IDX = 61

character_to_num[PAD_TOKEN] = PAD_TOKEN_IDX
character_to_num[START_TOKEN] = START_TOKEN_IDX
character_to_num[END_TOKEN] = END_TOKEN_IDX
num_to_character = {j: i for i, j in character_to_num.items()}


class TokenHashTable:
    def __init__(
        self, word2index_mapping=character_to_num, index2word_mapping=num_to_character
    ):
        self.word2index = word2index_mapping
        self.index2word = index2word_mapping

    def _indexesfromsentence(self, sentence):
        return [self.word2index[word] for word in sentence]

    def sentence_to_tensor(self, sentence):
        indexes = self._indexesfromsentence(sentence)
        return torch.tensor(indexes, dtype=torch.long)

    def index_to_sentence(self, indexes_list):
        if torch.is_tensor(indexes_list):
            indexes_list = indexes_list.tolist()
        words = [self.index2word[idx] for idx in indexes_list]
        return words


def read_file(file, file_id, landmarks_metadata_path):
    phrase_list = []
    frames_list = []
    metadata_train_dataframe = pd.read_csv(landmarks_metadata_path)
    file_id_df = metadata_train_dataframe.loc[
        metadata_train_dataframe["file_id"] == file_id
    ]
    saved_parueat_df = pq.read_table(
        file, columns=["sequence_id"] + FEATURE_COLUMNS
    ).to_pandas()
    for seq_id, phrase in zip(file_id_df.sequence_id, file_id_df.phrase):
        frames = saved_parueat_df[saved_parueat_df.index == seq_id].to_numpy()
        # NaN
        frames_list.append(torch.tensor(frames))
        phrase_list.append(phrase)
    return (frames_list, phrase_list)


class LandmarkDataset(Dataset):
    def __init__(self, file_path, file_id, table, transform=True):
        self.landmarks_metadata_path = METADATA
        self.frames, self.labels = read_file(
            file_path, file_id, self.landmarks_metadata_path
        )
        self.trans = transform
        self.table = table

    def _label_pre(self, label_sample):
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


def get_dataloader(file_path, file_id, batch_size=32, num_workers_=1):
    lookup_table = TokenHashTable(character_to_num, num_to_character)
    dataset = LandmarkDataset(file_path, file_id, lookup_table, transform=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers_,
        pin_memory=True,
    )
    return dataloader
