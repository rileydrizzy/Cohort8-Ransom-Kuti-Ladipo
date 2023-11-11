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
MODEL_DIR = ''
parquet_files = glob.glob(f"{LANDMARK_DIR}/*.parquet")
file_ids = [os.path.basename(file) for file in parquet_files]
assert len(parquet_files) == len(file_ids), "Failed Import of Files"
files = zip(parquet_files, file_ids)


def train(model, optim, loss_func, n_epochs, FILES, batch, device, validation=False):
    # To ensure reproducibility of the training process
    set_seed()
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        # Keeps track of the numbers of epochs
        # by updating the corresponding attribute
        total_epochs = epoch
        for file, file_id in FILES:
            dataloader = get_dataloader(file, file_id, batch_size=batch)
            # inner loop
            # Performs training using mini-batches
            loss = mini_batch(model, dataloader, optim, device, loss_func)
            train_losses.append(loss)
    save_checkpoint("dir", model, optim, total_epochs, train_losses, val_losses)


def mini_batch(model, dataloader, optim, loss_func, device, validation=False):
    # The mini-batch can be used with both loaders
    # The argument `validation`defines which loader and
    # corresponding step function is going to be used
    if validation:
        step_func = val_step_func
    else:
        step_func = train_step_func

    n_batches = len(dataloader)
    # Once the data loader and step function, this is the same
    # mini-batch loop we had before
    mini_batch_losses = []
    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = step_func(model, optim, loss_func, x_batch, y_batch)
        mini_batch_losses.append(loss)
    loss = np.mean(mini_batch_losses)
    return loss


def train_step_func():
    def perform_train_step_fn(model, optim, loss_func, x_batch, y_batch):
        model.train()
        preds = model(x_batch)
        loss = loss_func(preds, y_batch)
        loss.backward()
        optim.step()
        optim.zero_grad()
        return loss.item()

    return perform_train_step_fn


def val_step_func():
    def perform_val_step_fn(model, loss_func, x_batch, y_batch):
        model.eval()
        preds = model(x_batch)
        loss = loss_func(preds, y_batch)
        return loss.item()

    return perform_val_step_fn


def save_checkpoint(
    filename,
    model,
    optimizer,
    total_epochs,
    train_losses,
    val_losses
):
    # Builds dictionary with all elements for resuming training
    checkpoint = {
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_losses,
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
    logger.info("Starting training")

    DEVICE = get_device_strategy(tpu=False)
    logger.info(f"Trainig on {DEVICE}")

    model = ModelLoader().get_model(args)

    optizmer = optim.Adam(model.parameters(), args)
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    logger.info("f")
    wandb.init()

    train(
        model=args.model,
        optim=optizmer,
        loss_func=criterion,
        n_epochs=args.epochs,
        batch=args.batch,
        FILES=files,
        device=DEVICE,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
