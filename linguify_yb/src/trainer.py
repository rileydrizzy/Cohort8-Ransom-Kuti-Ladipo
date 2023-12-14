"""
Module for distributed training with PyTorch using Distributed Data Parallel (DDP).

Classes:
- Trainer: A class for training neural network models in a distributed setup.

Functions:
- ddp_setup: Setup Distributed Data Parallel (DDP) for training.
"""

# TODO Complete and refactor code for distributed training

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from utils.logger_util import logger


def ddp_setup():
    """
    Setup Distributed Data Parallel (DDP) for training.
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        loss_func,
    ):
        """
        Initialize a Trainer instance.

        Parameters:
            - model (torch.nn.Module): The neural network model.
            - train_data (DataLoader): The DataLoader for training data.
            - optimizer (torch.optim.Optimizer): The optimizer for training.
            - save_every (int): Save a snapshot of the model every `save_every` epochs.
            - loss_func: The loss function for training.
        """
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = "snapshot.pt"
        if os.path.exists(self.snapshot_path):
            logger.info("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        """
        Load a snapshot of the model.

        Parameters:
            - snapshot_path (str): Path to the snapshot file.
        """
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        logger.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        """
        Run a training batch.

        Parameters:
            - source: _type_
            - targets: _type_
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        """
        Run a training epoch.

        Parameters:
            - epoch (int): The current epoch.
        """
        b_sz = len(next(iter(self.train_data))[0])
        logger.info(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        """
        Save a snapshot of the model.

        Parameters:
            - epoch (int): The current epoch.
        """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        logger.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        """
        Train the model for a specified number of epochs.

        Parameters:
            - max_epochs (int): The maximum number of epochs to train.
        """
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
