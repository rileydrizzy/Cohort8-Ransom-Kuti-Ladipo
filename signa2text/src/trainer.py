"""
Module for distributed training with PyTorch using Distributed Data Parallel (DDP).

Classes:
- Trainer: A class for training neural network models in a distributed setup.

Functions:

"""

# TODO Implement and refactor code for distributed training
# TODO AMP Mixed Trainig and Scaled Loss
# TODO Consider adding TPU training

import os
import wandb
import torch

from utils.logging import logger


class Trainer:
    """_summary_"""

    def __init__(
        self,
        model,
        train_data,
        optimizer,
        loss_func,
        resume_checkpoint=False,
    ):
        """
        Initialize a Trainer instance.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model.
        train_data : DataLoader
            The DataLoader for training data.
        optimizer : torch.optim.Optimizer
            The optimizer for training.
        loss_func : _type_
            The loss function for training.
        resume_checkpoint : bool
            _description_
        """
        self.gpu_device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.epochs_run = 0
        self.snapshot_path = "snapshot.pt"
        if os.path.exists(self.snapshot_path):
            self._load_snapshot(snapshot_path=self.snapshot_path)
        elif resume_checkpoint:
            self._load_snapshot(from_wandb=True)

    def _load_snapshot(self, snapshot_path=None, from_wandb=False):
        """
        Load a snapshot of the model.

        Parameters
        ----------
        snapshot_path : str, optional
            Path to the snapshot file, by default None
        from_wandb : bool, optional
            _description_, by default False
        """
        logger.info("Loading snapshot")
        if from_wandb:
            # TODO load from wandb
            self.model = None
            self.epochs_run = None
        else:
            loc = "cuda:0"
            snapshot = torch.load(snapshot_path, map_location=loc)
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.epochs_run = snapshot["EPOCHS_RUN"]
        logger.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch, save_to_wandb=False):
        """
        Save a snapshot of the model.

        Parameters
        ----------
        epoch : int
            The current epoch.
        save_to_wandb : bool, optional
            _description_, by default False
        """

        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        if save_to_wandb:
            # TODO save on wandb
            pass
        logger.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

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
        batch_size = len(next(iter(self.train_data))[0])
        logger.info(
            f"[GPU{self.gpu_device}] Epoch {epoch} | Batchsize: \
                {batch_size} | Steps: {len(self.train_data)}"
        )
        # self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_device)
            targets = targets.to(self.gpu_device)
            self._run_batch(source, targets)

    def train(
        self,
        max_epochs: int,
        save_every,
        wandb_monitor=False,
    ):
        """
        Train the model for a specified number of epochs.

        Parameters
        ----------
        max_epochs : int
             The maximum number of epochs to train.
        save_every : int
            Save a snapshot of the model every `save_every` epochs.
        wandb_monitor : bool, optional
            _description_, by default False
        """

        if wandb_monitor:
            wandb.init(project="NSL_2_AUDIO")
            wandb.watch(self.model)
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if epoch % save_every == 0:
                self._save_snapshot(epoch, save_to_wandb=wandb_monitor)
