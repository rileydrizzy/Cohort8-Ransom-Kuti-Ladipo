"""
doc

# Usage:
# python -m src/train.py \
# --epochs 10 \
# --batch 512 \
"""
# TODO Complete and refactor code

import glob
import os

import numpy as np
import torch
import wandb
from torch import nn, optim

from linguify_yb.src.dataset.dataset import get_dataloader
from linguify_yb.src.models.model_loader import ModelLoader
from linguify_yb.src.utils import get_device_strategy, parse_args, set_seed
from linguify_yb.src.utils.logger_util import logger

try:
    LANDMARK_DIR = "data/raw/asl"
    MODEL_DIR = "model_dir"
    parquet_files = glob.glob(f"{LANDMARK_DIR}/*.parquet")
    file_ids = [os.path.basename(file) for file in parquet_files]
    assert len(parquet_files) == len(file_ids), "Failed Import of Files"
    TRAIN_FILES = list(zip(parquet_files, file_ids))
except AssertionError as asset_error:
    logger.info(f"fail {asset_error}")


def train(model, optim, loss_func, n_epochs, batch, device):
    # To ensure reproducibility of the training process
    set_seed()
    train_losses = []
    val_losses = []
    val_dataloader = get_dataloader(TRAIN_FILES[0], TRAIN_FILES[1], batch_size=batch)

    for epoch in range(n_epochs):
        # Keeps track of the numbers of epochs
        # by updating the corresponding attribute
        total_epochs = epoch
        file_train_loss = []
        for file, file_id in TRAIN_FILES:
            train_dataloader = get_dataloader(file, file_id, batch_size=batch)

            # Performs training using mini-batches
            train_loss = mini_batch(
                model, train_dataloader, optim, device, loss_func, validation=False
            )
            file_train_loss.append(train_loss)
            train_loss = np.mean(file_train_loss)
            train_losses.append(train_loss)

        # Performs evaluation using mini-batches
        with torch.no_grad():
            val_loss = mini_batch(
                model, val_dataloader, optim, loss_func, device, validation=True
            )
            val_losses.append(val_loss)

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
            }
        )
        # Checkpoint model
        if epoch // 2 == 0:
            save_checkpoint(
                MODEL_DIR, model, optim, total_epochs, train_losses, val_losses
            )


def mini_batch(model, dataloader, optim, loss_func, device, validation=False):
    # The mini-batch can be used with both loaders
    # The argument `validation`defines which loader and
    # corresponding step function is going to be used
    if validation:
        step_func = val_step_func()
    else:
        step_func = train_step_func()

    # Once the data loader and step function, this is the same
    # mini-batch loop we had before
    mini_batch_losses = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = step_func(model, optim, loss_func, x=x_batch, y=y_batch)
        mini_batch_losses.append(loss)
    loss = np.mean(mini_batch_losses)
    return loss


def train_step_func():
    def perform_train_step_fn(model, optim, loss_func, x, y):
        model.train()
        preds = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        return loss.item()

    return perform_train_step_fn


def val_step_func():
    def perform_val_step_fn(
        model,
        loss_func,
        x,
        y,
    ):
        model.eval()
        preds = model(x)
        loss = loss_func(preds, y)
        return loss.item()

    return perform_val_step_fn


def save_checkpoint(filename, model, optimizer, total_epochs, train_losses, val_losses):
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


def main(arg):
    logger.info("Starting training")

    DEVICE = get_device_strategy(tpu=arg.tpu)
    logger.info(f"Trainig on {DEVICE}")

    model = ModelLoader().get_model(arg.model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)

    # Optimizes given model/function using TorchDynamo and specified backend
    torch.complie(model)
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
        logger.info("finished")
    except Exception as error:
        logger.exception(f"Trainig failed{error}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
