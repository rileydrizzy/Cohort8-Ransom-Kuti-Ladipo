"""
doc
# Usage:

#torchrun --standalone \
#--nproc_per_node=<NUM_GPUS>\
src/main.py \
# --epochs 10 \
# --batch 512 \
# python -m src/main.py --epochs 10 --batch 512
"""
#TODO complete documentation
# TODO Complete and refactor code for distributed training

import os
import json

import numpy as np
import torch
import wandb
from torch import nn

from utils.util import parse_args, set_seed
from utils.logger_util import logger
from models.model_loader import ModelLoader
from dataset.dataset_loader import get_dataset, prepare_dataloader
from trainer import Trainer, ddp_setup, destroy_process_group()

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


def load_train_objs():
    model = ModelLoader().get_model(arg.model)
    # Optimizes given model/function using TorchDynamo and specified backend
    torch.compile(model)
    optimizer_ = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    dataset = get_dataset()
    return model, optimizer_, dataset, criterion

def main(arg, save_every: int, total_epochs: int, batch_size: int):
    logger.info(f"Starting training on {arg.model}")
    # To ensure reproducibility of the training process
    set_seed()
    logger.info(f"Training on {DEVICE} for {arg.epochs} epochs.")

    try:
        ddp_setup()
        dataset, model, optimizer, criterion = load_train_objs()
        train_dataset = prepare_dataloader(dataset, arg.batch_size, )
        trainer = Trainer(
            model=model,
            train_data=train_dataset,
            optimizer=optimizer,
            save_every=2,
            loss_func=criterion,
        )

        trainer.train(total_epochs)
        destroy_process_group()

        logger.success(f"Training completed: {arg.epochs} epochs on {DEVICE}.")
    except Exception as error:
        logger.exception(f"Training failed due to an {error}.")


if __name__ == "__main__":
    parse_args
    main()
