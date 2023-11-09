"""doc
"""

import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader, Dataset
from linguify_yb.src.utils.logger_util import logger
from linguify_yb.src.utils import set_seed

SEED = 42
set_seed(SEED)

