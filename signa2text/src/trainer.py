"""
Module for distributed training with

Functions:

"""

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.profilers import SimpleProfiler

# from utils.logging import logger

# Checkpoint Filename Template
FILENAME_TEMPLATE = "NSL-2-AUDIO-{epoch}-{val_loss:.2f}"

profiler = SimpleProfiler(dirpath=".", filename="perf_logs")


class LitModule(L.LightningModule):
    """_summary_"""

    def __init__(
        self, model_name, model, loss_criterion, acc_metric, save_ckpt_every=5
    ):
        super().__init__()
        self.model = model
        self.loss_criterion = loss_criterion
        self.accuracy_metric = acc_metric
        self.save_ckpt_every = save_ckpt_every
        self.checkpoint_dir = f"artifacts/{model_name}/"
        self.save_hyperparameters()

    def _get_preds_loss_accuracy(self, batch):
        sources, targets = batch
        preds = self.model(sources, targets)
        # loss = self.loss_criterion(preds, targets)
        
        acc_loss = self.accuracy_metric()
        return loss, acc_loss, preds

    def training_step(self, batch, batch_idx):
        loss, acc_loss, preds = self._get_preds_loss_accuracy(batch)

        self.log("loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, _, preds = self._get_preds_loss_accuracy(batch)

        self.log("val_loss", val_loss, on_epoch=True, on_step=False)
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
