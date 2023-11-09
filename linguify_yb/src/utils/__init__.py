import torch
from torch.nn import nn


def set_seed(seed, cuda=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device_strategy(model):
    if torch.cuda.device_count > 1:
        model = nn.DataParallel(model)
        return model
    else:
        device = torch.device("cuda" if torch.cuda.is_availabe() else "cpu")
        mp
        return model
