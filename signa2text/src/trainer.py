"""
Module for distributed training with

Functions:

"""

# TODO implement loss and metrics

import torch
import lightning as L

# from utils.logging import logger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.profilers import SimpleProfiler

# from metrics import NormalizedLevenshteinDistance

PROJECT_NAME = "NSL_2_AUDIO"

# Checkpoint Filename Template
FILENAME_TEMPLATE = "NSL-2-AUDIO-{epoch}-{val_loss:.2f}"

profiler = SimpleProfiler(dirpath=".", filename="perf_logs")


class LitModule(L.LightningModule):
    """_summary_"""

    def __init__(self, model, loss_criterion, metric, save_ckpt_every=5, model_name="test"):
        super().__init__()
        self.model = model
        self.loss_criterion = loss_criterion
        self.metric = metric
        self.save_ckpt_every = save_ckpt_every
        self.checkpoint_dir = f"artifacts/{model_name}/"
        self.save_hyperparameters()

    def _get_preds_loss_accuracy(self, batch):
        sources, targets = batch
        preds = self.model(sources, targets)
        loss = self.loss_criterion(preds, sources)
        # Levenshtein_dis = self.metric(oi)
        return preds, loss

    def training_step(self, batch, batch_idx):
        preds, loss = self._get_preds_loss_accuracy(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, val_loss = self._get_preds_loss_accuracy(batch)
        self.log("val_loss", val_loss)
        return preds

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
            every_n_epochs=self.save_ckpt_every,
            save_on_train_epoch_end=False,
        )
        device_callback = DeviceStatsMonitor(cpu_stats=True)

        callbacks_list = [checkpoint_callback, early_stopping_callback, device_callback]
        return callbacks_list
