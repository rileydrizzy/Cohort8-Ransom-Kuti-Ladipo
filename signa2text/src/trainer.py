"""
Module for distributed training with PyTorch using Distributed Data Parallel (DDP).

Classes:
- Trainer: A class for training neural network models in a distributed setup.

Functions:

"""

# TODO Implement and refactor code for distributed training on Multi-GPU
# TODO Consider adding TPU training using PyTorch Lightning
# TODO implement loss and metrics

from pathlib import Path
import torch
import lightning as L
import wandb

# from utils.logging import logger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler

# from metrics import NormalizedLevenshteinDistance

PROJECT_NAME = "NSL_2_AUDIO"

# Checkpoint Filename Template
FILENAME_TEMPLATE = "NSL-2-AUDIO-{epoch}-{val_loss:.2f}"

MAX_TRAIN_TIME = "00:06:00:00"

wandb_logger = WandbLogger(project=PROJECT_NAME, log_model="all")

profiler = SimpleProfiler(dirpath=".", filename="perf_logs")


class LitModule(L.LightningModule):
    """_summary_"""

    def __init__(self, model, loss_criterion, metric, model_name="test"):
        super().__init__()
        self.model = model
        self.loss_criterion = loss_criterion
        self.metric = metric
        self.checkpoint_dir = f"artifact/{model_name}/"
        wandb_logger.watch(model, log="all")
        self.save_hyperparameters()

    def resume_training_(self, from_wandb=False, MODEL_RUN_ID=None, VERSION="latest"):
        if from_wandb:
            # reference can be retrieved in artifacts panel
            # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
            checkpoint_reference = (
                f"rileydrizzy/{PROJECT_NAME}/{MODEL_RUN_ID}:{VERSION}"
            )
            # download checkpoint locally (if not already cached)
            run = wandb.init(project=PROJECT_NAME)
            artifact = run.use_artifact(checkpoint_reference, type="model")
            artifact_dir = artifact.download()
            model_checkpoint = Path(artifact_dir) / "model.ckpt"
            self.on_load_checkpoint(model_checkpoint)

    def _get_preds_loss_accuracy(self, batch):
        source, targets = batch
        pred_outputs = self.model(source, targets)
        loss = self.loss_criterion(pred_outputs, source)
        # Levenshtein_dis = self.metric(oi)
        return pred_outputs, loss

    def training_step(self, batch, batch_idx):
        preds, loss = self._get_preds_loss_accuracy(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, val_loss = self._get_preds_loss_accuracy(batch)
        self.log("val_loss", val_loss)
        return preds

    """
    def on_train_epoch_end():
        pass
    """

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 5,
            },
        }

    def configure_callbacks(self):
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", check_on_train_epoch_end=False, patience=3
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=FILENAME_TEMPLATE,
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        )
        device_callback = DeviceStatsMonitor(cpu_stats=True)

        callbacks_list = [checkpoint_callback, early_stopping_callback, device_callback]
        return callbacks_list


trainer = L.Trainer(
    accelerator="auto",
    precision="16-mixed",
    logger=wandb_logger,
    profiler=profiler,
    strategy="auto",
    check_val_every_n_epoch=5,
    max_time=MAX_TRAIN_TIME,
    max_epochs=10,
)
