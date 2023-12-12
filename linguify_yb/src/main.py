"""
doc

# Usage:
# python -m src/train.py \
# --epochs 10 \
# --batch 512 \
"""
# TODO Complete and refactor code for distributed training

import os
import json

import numpy as np
import torch
import wandb
from torch import nn

from utils.util import get_device_strategy, parse_args, set_seed
from utils.logger_util import logger
from models.model_loader import ModelLoader
from dataset.dataset_loader import get_dataloader
import trainer

try:
    dataset_paths = "data/dev_samples.json"  # On kaggle replace with "data/dataset_paths.json" to train on full data
    with open(dataset_paths, "r", encoding="utf-8") as json_file:
        data_dict = json.load(json_file)
    LANDMARK_DIR = "/kaggle/input/asl-fingerspelling/train_landmarks"
    MODEL_DIR = "model.pt"

    # Training dataset
    train_dataset = data_dict["train_files"]
    train_file_ids = [os.path.basename(file) for file in train_dataset]
    train_file_ids = [
        int(file_name.replace(".parquet", "")) for file_name in train_file_ids
    ]
    assert len(train_dataset) == len(
        train_file_ids
    ), "Failed import of Train files path "
    TRAIN_DS_FILES = list(zip(train_dataset, train_file_ids))

    # Validation dataset
    valid_dataset = data_dict["valid_files"]
    valid_file_ids = [os.path.basename(file) for file in valid_dataset]
    valid_file_ids = [
        int(file_name.replace(".parquet", "")) for file_name in valid_file_ids
    ]
    assert len(train_dataset) == len(
        train_file_ids
    ), "Failed Import of Valid Files path"
    VALID_DS_FILES = list(zip(valid_dataset, valid_file_ids))
except AssertionError as asset_error:
    logger.exception(f"failed {asset_error}")


def main(arg):
    logger.info(f"Starting training on {arg.model}")
    # To ensure reproducibility of the training process
    set_seed()
    DEVICE = get_device_strategy(tpu=arg.tpu)
    logger.info(f"Training on {DEVICE} for {arg.epochs} epochs.")

    model = ModelLoader().get_model(arg.model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizes given model/function using TorchDynamo and specified backend
    torch.compile(model)

    logger.info("training")
    wandb.init(
        project="ASL-project",
        config={
            "learning_rate": 0.01,
            "architecture": "Test Model",
            "dataset": "Google ASL Landmarks",
            "epochs": 12,
        },
    )

    wandb.watch(model)
    try:
        train(
            model=arg.model,
            optim=optimizer,
            loss_func=criterion,
            n_epochs=arg.epochs,
            batch=arg.batch,
            device=DEVICE,
        )
        logger.success(f"Training completed: {arg.epochs} epochs on {DEVICE}.")

    except Exception as error:
        logger.exception(f"Training failed due to an {error}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
