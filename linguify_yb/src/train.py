"""doc
"""
# TODO Complete and refactor code

import glob
import os

import torch
import torch.nn as nn
import wandb
from torch import optim

from linguify_yb.src.dataset.dataset import get_dataloader
from linguify_yb.src.models.model_loader import ModelLoader
from linguify_yb.src.utils import get_device_strategy, set_seed
from linguify_yb.src.utils.logger_util import logger

# from linguify_yb.src.utils.args import

LANDMARK_DIR = "data/raw/asl"
parquet_files = glob.glob(f"{LANDMARK_DIR}/*.parquet")
file_ids = [os.path.basename(file) for file in parquet_files]
assert len(parquet_files) == len(file_ids), "Failed"

SEED = 42
set_seed(SEED)
LEARNING_RATE = 0
NUM_EPOCHS = 10


dataloader = get_dataloader()
model_loader = ModelLoader()
model = model_loader.get_model("name")
optizmer = optim.Adam(model.parameters(), LEARNING_RATE)
criterion = nn


for epoch in range(NUM_EPOCHS):
    for frames_x, phrase_y in dataloader:
        frames_x = frames_x.to(device)
        phrase_y = phrase_y.to(device)
        optizmer.zero_grad()
        outputs = model(frames_x)
        loss = criterion(outputs, phrase_y)
        loss.backward()
        optizmer.step()
