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

from utils.logger_util import logger


def train(model, optim, loss_func, n_epochs, batch, device,):

    model.to(device)
    
    train_losses = []
    val_losses = []
    val_dataloader = # get_dataloader(TRAIN_FILES[0][0], TRAIN_FILES[0][1], batch_size=batch)
    for epoch in range(n_epochs):
        logger.info(f"Training on epoch {epoch}.")
        total_epochs = epoch
        file_train_loss = []
        for file, file_id in TRAIN_DS_FILES:
            train_dataloader =  # get_dataloader(file, file_id, batch_size=batch)

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


def distributed_stra_gpu():
    pass
