import torch

def set_seed(seed, cuda=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_strategy():
    pass