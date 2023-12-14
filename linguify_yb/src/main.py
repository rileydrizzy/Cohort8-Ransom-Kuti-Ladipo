"""
doc
# Usage:

#torchrun --standalone \
#--nproc_per_node=<NUM_GPUS>\
src/main.py \
# --epochs 10 \
# --batch 512 \
# python -m src/main.py --epochs 10 --batch 512
"""
# TODO cleanup and complete documentation
# TODO Complete and refactor code for distributed training
# TODO remove test model and test data
import os
import json
import torch

from torch import nn

from utils.util import parse_args, set_seed
from utils.logger_util import logger
from models.model_loader import ModelLoader
from dataset.dataset_loader import get_dataset, prepare_dataloader, get_test_dataset
from trainer import Trainer, ddp_setup
from torch.distributed import destroy_process_group

try:
    # On kaggle replace with "data/dataset_paths.json" to train on full data
    DATASET_PATHS = "data/dev_samples.json"
    with open(DATASET_PATHS, "r", encoding="utf-8") as json_file:
        dataset_paths_dict = json.load(json_file)

    # Training dataset
    train_dataset_dict = dataset_paths_dict["train_files"]
    train_file_ids = [os.path.basename(file) for file in train_dataset_dict]
    train_file_ids = [
        int(file_name.replace(".parquet", "")) for file_name in train_file_ids
    ]
    assert len(train_dataset_dict) == len(
        train_file_ids
    ), "Failed getting Train files path"
    TRAIN_DS_FILES = list(zip(train_dataset_dict, train_file_ids))

    # Validation dataset
    valid_dataset_dict = dataset_paths_dict["valid_files"]
    valid_file_ids = [os.path.basename(file) for file in valid_dataset_dict]
    valid_file_ids = [
        int(file_name.replace(".parquet", "")) for file_name in valid_file_ids
    ]
    assert len(train_dataset_dict) == len(
        train_file_ids
    ), "Failed getting of Valid Files path"
    VALID_DS_FILES = list(zip(valid_dataset_dict, valid_file_ids))
except AssertionError as asset_error:
    logger.exception(f"failed due to {asset_error}")


def load_train_objs(model_name, files):
    model = ModelLoader().get_model(model_name)

    # Optimizes given model/function using TorchDynamo and specified backend
    torch.compile(model)
    optimizer_ = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    dataset = get_test_dataset()  # get_dataset(files)
    return model, optimizer_, dataset, criterion


def main(model_name: str, save_every: int, total_epochs: int, batch_size: int):
    logger.info(f"Starting training on {model_name}, epoch -> {total_epochs}")
    logger.info(f"Batch Size -> {batch_size}, model saved every -> {save_every} epoch")

    # To ensure reproducibility of the training process
    set_seed()

    try:
        ddp_setup()
        dataset, model, optimizer, criterion = load_train_objs(
            model_name, files=TRAIN_DS_FILES
        )
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
        logger.exception(f"Training failed due to an {error}.")


if __name__ == "__main__":
    arg = parse_args()
    logger.info(f"{arg.model_name}")
    main(
        model_name=arg.model_name,
        save_every=arg.save_every,
        total_epochs=arg.epochs,
        batch_size=arg.batch,
    )
