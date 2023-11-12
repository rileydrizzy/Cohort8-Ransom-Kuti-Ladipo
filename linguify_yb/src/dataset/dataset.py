"""doc
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from linguify_yb.src.dataset.frames_config import FEATURE_COLUMNS, LHAND_IDX, RHAND_IDX
from linguify_yb.src.dataset.preprocess import frames_preprocess

PHRASE_PATH = "/kaggle/input/asl-fingerspelling/character_to_prediction_index.json"
METADATA = "data/raw/train.csv"

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


class StaticHashTable:
    def __init__(self, word2index_mapping, index2word_mapping):
        self.word2index = word2index_mapping
        self.index2word = index2word_mapping

    def _indexesfromsentence(self, sentence):
        return [self.word2index[word] for word in sentence]

    def tensorfromsentence(self, sentence):
        indexes = self._indexesfromsentence(sentence)
        return torch.tensor(indexes, dtype=torch.long)  # .view(1, -1)


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
        right_num_nan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
        left_num_nan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
        total_num_nan = max(right_num_nan, left_num_nan)
        if 2 * len(phrase) < total_num_nan:
            frames_list.append(frames)
            phrase_list.append(phrase)
    return (frames_list, phrase_list)


class LandmarkDataset(Dataset):
    def __init__(self, file_path, file_id, table, transform=False):
        self.landmarks_metadata_path = METADATA
        self.frames, self.labels = read_file(
            file_path, file_id, self.landmarks_metadata_path
        )
        self.trans = transform
        self.table = table

    def _label_pre(self, label_sample):
        sample = START_TOKEN + label_sample + END_TOKEN
        new_phrase = self.table.tensorFromSentence(list(sample))
        ans = torch.nn.functional.pad(
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
            frames = frames_preprocess(frames)
        return frames, phrase


def pack_collate_func(batch):
    frames_feature = [item[0] for item in batch]
    phrase = [item[1] for item in batch]
    return frames_feature, phrase


def get_dataloader(file_path, file_id, batch_size):
    lookup_table = StaticHashTable(character_to_num, num_to_character)
    dataset = LandmarkDataset(file_path, file_id, lookup_table, transform=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        collate_fn=pack_collate_func,
        pin_memory=True,
    )
    return dataloader


# Test training pipeline
class TestDataset(Dataset):
    """test"""
    def __init__(self, num_samples=1000, input_size=10):
        self.num_samples = num_samples
        self.input_size = input_size
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_test_dataloader():
    # Generating a dataset with 1000 samples and 10 input features
    dataset = TestDataset(num_samples=1000, input_size=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
