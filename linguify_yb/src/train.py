"""doc
"""
# TODO Complete and refactor code

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import optim

from linguify_yb.src.dataset.dataset import get_dataloader
from linguify_yb.src.models.model_loader import ModelLoader
from linguify_yb.src.utils import get_device_strategy, set_seed
from linguify_yb.src.utils.args import parse_args
from linguify_yb.src.utils.logger_util import logger

LANDMARK_DIR = "data/raw/asl"
parquet_files = glob.glob(f"{LANDMARK_DIR}/*.parquet")
file_ids = [os.path.basename(file) for file in parquet_files]
assert len(parquet_files) == len(file_ids), "Failed Import of Files"
files = zip(parquet_files, file_ids)


def train(model, optim, loss_func, n_epochs, FILES, BATCH, device):
    # To ensure reproducibility of the training process
    set_seed()

    for epoch in range(n_epochs):
        # Keeps track of the numbers of epochs
        # by updating the corresponding attribute
        total_epochs += 1
        for file, file_id in FILES:
            dataloader = get_dataloader(file, file_id, batch_size=BATCH)
            # inner loop
            # Performs training using mini-batches
            mini_batch(
                model,
                dataloader,
                device,
            )


def mini_batch(model, dataloader, optim, device, validation=False):
    # The mini-batch can be used with both loaders
    # The argument `validation`defines which loader and
    # corresponding step function is going to be used

    n_batches = len(dataloader)
    # Once the data loader and step function, this is the same
    # mini-batch loop we had before
    mini_batch_losses = []
    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        model.train()
        preds = model(x_batch)
        loss = loss_func(preds, y_batch)
        loss.backward()
        optim.step()
        optim.zero_grad()
        mini_batch_losses.append(loss.item())
        return


def save_checkpoint(
    filename,
    model,
    optimizer,
    total_epochs,
):
    # Builds dictionary with all elements for resuming training
    checkpoint = {
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses,
        "val_loss": val_losses,
    }

    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    # Loads dictionary
    checkpoint = torch.load(filename)

    # Restore state for model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    total_epochs = checkpoint["epoch"]
    losses = checkpoint["loss"]
    val_losses = checkpoint["val_loss"]
    return model


def main(args):
    logger.info(f"Starting training")

    DEVICE = get_device_strategy()
    logger.info(f"Trainig on {DEVICE}")

    model = ModelLoader().get_model(args)
    if args:  # LOAD MODEL PARA,
        model = load_checkpoint()

    optizmer = optim.Adam(model.parameters(), args)
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    logger.info("f")
    wandb.init()

    train(
        model=model,
        optim=optizmer,
        loss_func=criterion,
        n_epochs=2,
        BATCH=args,
        FILES=files,
        device=DEVICE,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
