"""
Module for distributed training with PyTorch using Distributed Data Parallel (DDP).

"""

# TODO cleanup and complete documentation
# TODO Complete and refactor code for distributed training

import torch
import hydra

from omegaconf import DictConfig
from utils.tools import resume_training, set_seed
from utils.logging import logger
from models.model_loader import ModelLoader
from dataset.dataset_loader import get_dataset, prepare_dataloader  # get_test_dataset
from dataset.dataset_paths import get_dataset_paths
from lightning import Trainer
from trainer import LitModule, PROJECT_NAME, profiler

from lightning.pytorch.loggers import WandbLogger


MAX_TRAIN_TIME = "00:06:00:00"


def initialize_wandb(model, run_id, resume_arg="allow"):
    if run_id != "None":
        wandb_logger = WandbLogger(
            project=PROJECT_NAME, log_model="all", id=run_id, resume=resume_arg
        )
    else:
        wandb_logger = WandbLogger(project=PROJECT_NAME, log_model="all")
    wandb_logger.watch(model, log="all")
    return wandb_logger


def load_train_objs(model_name):
    """
    Load training objects, including the model, optimizer, dataset, and criterion.

    Parameters
    ----------
    model_name : str
        Name of the model to be loaded.
    files_paths :
        Optional parameter for specifying files.

    Returns
    -------
    _type_
        model: The loaded model optimizer_: The optimizer for training.
        dataset: The training dataset criterion: The loss criterion for training.
    """
    model = ModelLoader().get_model(model_name)

    # Optimizes given model/function using TorchDynamo and specified backend
    torch.compile(model)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    return model, criterion


@hydra.main(config_name="train", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function for training a model.
    """
    logger.info(f"Model to be trained is: {cfg.model_name}")
    logger.info(
        f"Starting training on {cfg.model_name}, epoch -> {cfg.params.total_epochs}"
    )
    logger.info(
        f"Batch Size -> {cfg.params.batch_size}, model to be saved every -> {cfg.params.save_every} epoch"
    )
    training_status = None
    try:
        # To ensure reproducibility of the training process
        set_seed()

        train_data_paths, valid_data_paths = get_dataset_paths(dev_mode=cfg.dev_mode)

        model, criterion = load_train_objs(cfg.model_name)

        logger.info("Initializing WANDB")

        wandb_logger = initialize_wandb(model, run_id=cfg.wandb_params.model_run_id)

        checkpoint_path = None
        if cfg.wandb_params.resume_checkpoint:
            logger.info("Resuming training")
            checkpoint_path = resume_training(
                wandb_logger,
                project_name=PROJECT_NAME,
                model_run_id=cfg.wandb_params.model_run_id,
                version=cfg.wandb_params.model_version,
            )
        train_dataset = get_dataset(train_data_paths)
        train_dataset = prepare_dataloader(
            train_dataset,
            cfg.params.batch_size,
        )
        valid_dataset = get_dataset(valid_data_paths)
        valid_dataset = prepare_dataloader(
            valid_dataset,
            cfg.params.batch_size,
        )

        model = LitModule(
            model_name=cfg.model_name,
            model=model,
            loss_criterion=criterion,
            metric=None,
            save_ckpt_every=cfg.params.save_every,
        )
        trainer = Trainer(
            accelerator="auto",
            precision="16-mixed",
            logger=wandb_logger,
            profiler=profiler,
            strategy="auto",
            check_val_every_n_epoch=cfg.params.valid_epoch,
            max_time=MAX_TRAIN_TIME,
            max_epochs=10,
        )
        logger.info("Train")
        trainer.fit(
            model=model,
            train_dataloaders=train_dataset,
            val_dataloaders=None,
            ckpt_path=checkpoint_path,
        )
        logger.success(f"Training completed: {cfg.params.total_epochs} epochs.")
        training_status = "Success"

    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")
        training_status = "Failed"
    finally:
        wandb_logger.finalize(status=training_status)


if __name__ == "__main__":
    main()
