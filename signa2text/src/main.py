"""
Module for distributed training with PyTorch using Distributed Data Parallel (DDP).

"""

# TODO cleanup and complete documentation
# TODO Complete and refactor code for distributed training
# TODO remove test model and test data\
# TODO add wandb for monitoring and saving model state

import torch
import hydra

from omegaconf import DictConfig
from utils.tools import parse_args, set_seed
from utils.logging import logger
from models.model_loader import ModelLoader
from dataset.dataset_loader import get_dataset, prepare_dataloader  # get_test_dataset
from dataset.dataset_paths import get_dataset_paths

from trainer import Trainer

# from torch.distributed import destroy_process_group


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

    optimizer_ = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    return (
        model,
        optimizer_,
        criterion,
    )


def main(
    model_name: str,
    save_every: int,
    total_epochs: int,
    batch_size,
):
    """
    Main function for training a model.

    Parameters:
        - model_name (str): Name of the model to be trained.
        - save_every (int): Frequency of saving the model during training.
        - total_epochs (int): Total number of training epochs.
        - batch_size (int): Batch size for training.
    """
    logger.info(f"Starting training on {model_name}, epoch -> {total_epochs}")
    logger.info(
        f"Batch Size -> {batch_size}, model to be saved every -> {save_every} epoch"
    )

    try:
        # To ensure reproducibility of the training process
        set_seed()

        train_paths, valid_paths = get_dataset_paths(dev_mode=True)

        model, optimizer, criterion = load_train_objs(model_name)

        train_dataset = get_dataset(train_paths)  # get_test_dataset()
        train_dataset = prepare_dataloader(
            train_dataset,
            batch_size,
        )
        valid_dataset = get_dataset(valid_paths)  # get_test_dataset()
        valid_dataset = prepare_dataloader(
            valid_dataset,
            batch_size,
        )

        trainer = Trainer(
            model=model,
            train_data=train_dataset,
            optimizer=optimizer,
            loss_func=criterion,
            resume_checkpoint=False,
        )

        trainer.train(total_epochs, save_every=save_every, wandb_monitor=False)

        logger.success(f"Training completed: {total_epochs} epochs .")

    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")


if __name__ == "__main__":
    arg_ = parse_args()
    logger.info(f"Model to be trained is: {arg_.model_name}")
    main(
        model_name=arg_.model_name,
        save_every=arg_.save_every,
        total_epochs=arg_.epochs,
        batch_size=arg_.batch,
    )
