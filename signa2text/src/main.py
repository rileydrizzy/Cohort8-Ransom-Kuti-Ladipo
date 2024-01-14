"""
Module for distributed training with PyTorch using Distributed Data Parallel (DDP).

"""

# TODO cleanup and complete documentation
# TODO Complete and refactor code for distributed training
# TODO remove test model and test data\
# TODO add wandb for monitoring and saving model state

import torch

from torch import nn

from utils.util import parse_args, set_seed
from utils.logging import logger
from models.model_loader import ModelLoader
from dataset.dataset_loader import get_dataset, prepare_dataloader, get_test_dataset
from dataset.dataset_paths import get_dataset_paths
from trainer import Trainer, ddp_setup
from torch.distributed import destroy_process_group


def load_train_objs(model_name, files=None):
    """
    Load training objects, including the model, optimizer, dataset, and criterion.

    Parameters:
        - model_name (str): Name of the model to be loaded.
        - files: Optional parameter for specifying files.

    Returns:
        - model: The loaded model.
        - optimizer_: The optimizer for training.
        - dataset: The training dataset.
        - criterion: The loss criterion for training.
    """
    model = ModelLoader().get_model(model_name)

    # Optimizes given model/function using TorchDynamo and specified backend
    torch.compile(model)
    optimizer_ = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    dataset = get_test_dataset()  # get_dataset(files)
    return model, optimizer_, dataset, criterion


def main(model_name: str, save_every: int, total_epochs: int, batch_size: int):
    """
    Main function for training a model.

    Parameters:
        - model_name (str): Name of the model to be trained.
        - save_every (int): Frequency of saving the model during training.
        - total_epochs (int): Total number of training epochs.
        - batch_size (int): Batch size for training.
    """
    logger.info(f"Starting training on {model_name}, epoch -> {total_epochs}")
    logger.info(f"Batch Size -> {batch_size}, model saved every -> {save_every} epoch")

    # To ensure reproducibility of the training process
    set_seed()

    try:
        # train, valid = get_dataset_paths()
        ddp_setup()
        dataset, model, optimizer, criterion = load_train_objs(model_name)
        train_dataset = prepare_dataloader(
            dataset,
            batch_size,
        )
        trainer = Trainer(
            model=model,
            train_data=train_dataset,
            optimizer=optimizer,
            save_every=save_every,
            loss_func=criterion,
        )

        trainer.train(total_epochs)
        destroy_process_group()

        logger.success(f"Training completed: {total_epochs} epochs on.")
    except Exception as error:
        logger.exception(f"Training failed due to -> {error}.")


if __name__ == "__main__":
    arg = parse_args()
    logger.info(f"{arg.model_name}")
    main(
        model_name=arg.model_name,
        save_every=arg.save_every,
        total_epochs=arg.epochs,
        batch_size=arg.batch,
    )
