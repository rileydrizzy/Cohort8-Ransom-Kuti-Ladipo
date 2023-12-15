import os
import random
import argparse

import numpy as np
import torch

#import torch_xla.core.xla_model as xm

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device_strategy(tpu=False):
    if tpu:
        device = None #xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_availabe() else "cpu")
    return device


def parse_args():
    """
    Parse arguments given to the script.

    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description="Run distributed data-parallel training and log with wandb."
    )

    parser.add_argument(
        "--model_name",
        default="baseline_transfomer",
        type=str,
        metavar="N",
        help="name of model to train",
    )

    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch",
        default=32,
        type=int,
        metavar="N",
        help="number of data samples in one batch",
    )
    parser.add_argument(
        "--tpu",
        default=False,
        type=bool,
        metavar="N",
        help="Train on TPU Device",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=bool,
        help="Path to the checkpoint for resuming training",
    )
    parser.add_argument(
        "--save_every",
        default= 2,
        type=int,
        help="",
    )

    args = parser.parse_args()
    return args
