"""doc
"""

# TODO add loss/criterion for training
# TODO add Leve distances metric for eval


import torch
from torchmetrics.text import EditDistance


class NormalizedLevenshteinDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.levenshte_indistance = EditDistance(reduction="sum")

    def forward(self, predictions, targets):
        total_chars = sum(len(char) for char in targets)
        total_distance = self.levenshte_indistance(predictions, targets)
        result = (total_chars - total_distance) / total_chars
        return result


