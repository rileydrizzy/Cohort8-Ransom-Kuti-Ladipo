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

from linguify_yb.src.dataset.dataset import get_dataloader, TEST_LOADER
from linguify_yb.src.models.model_loader import ModelLoader
from linguify_yb.src.utils import get_device_strategy, parse_args, set_seed
from linguify_yb.src.utils.logger_util import logger


try:
    dataset_paths = "dev_samples.json"  # On kaggle replace with "dataset_paths.json" to train on full data
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


def train(model, optim, loss_func, n_epochs, batch, device):
    # To ensure reproducibility of the training process
    set_seed()
    train_losses = []
    val_losses = []
    val_dataloader = TEST_LOADER  # get_dataloader(TRAIN_FILES[0][0], TRAIN_FILES[0][1], batch_size=batch)

    for epoch in range(n_epochs):
        logger.info(f"Training on epoch {epoch}.")
        total_epochs = epoch
        file_train_loss = []
        for file, file_id in TRAIN_DS_FILES:
            train_dataloader = (
                TEST_LOADER  # get_dataloader(file, file_id, batch_size=batch)
            )

            # Performs training using mini-batches
            train_loss = mini_batch(
                model, train_dataloader, optim, loss_func, device, validation=False
            )
            file_train_loss.append(train_loss)
        train_loss = np.mean(file_train_loss)
        train_losses.append(train_loss)

        # Performs evaluation using mini-batches
        logger.info("Starting validation.")
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

        if epoch // 2 == 0:
            logger.info("Initiating checkpoint. Saving model and optimizer states.")
            save_checkpoint(
                MODEL_DIR, model, optim, total_epochs, train_losses, val_losses
            )


def mini_batch(
    model, dataloader, mini_batch_optim, loss_func, device, validation=False
):
    # The mini-batch can be used with both loaders
    # The argument `validation`defines which loader and
    # corresponding step function is going to be used
    if validation:
        step_func = val_step_func(model, loss_func)
    else:
        step_func = train_step_func(model, mini_batch_optim, loss_func)

    # Once the data loader and step function, this is the same
    # mini-batch loop we had before
    mini_batch_losses = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = step_func(x=x_batch, y=y_batch)
        mini_batch_losses.append(loss)
    loss = np.mean(mini_batch_losses)
    return loss


def train_step_func(model, optim_, loss_func):
    def perform_train_step_fn(x, y):
        model.train()
        preds = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        optim_.step()
        optim_.zero_grad()
        return loss.item()

    return perform_train_step_fn


def val_step_func(model, loss_func):
    def perform_val_step_fn(x, y):
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
    logger.info(f"Starting training on {arg.model}")

    DEVICE = get_device_strategy(tpu=arg.tpu)
    logger.info(f"Training on {DEVICE} for {arg.epochs} epochs.")

    model = ModelLoader().get_model(arg.model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = model.to(DEVICE)

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
