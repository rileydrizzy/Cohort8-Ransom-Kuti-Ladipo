"""doc
"""

# TODO add loss/criterion for training
# TODO add Lv distances metric for eval
# TODO add CTC Loss


import torch
from torchmetrics import Metric


# impl loss
"""
In summary, "normalized total Levenshtein distance" adjusts the raw Levenshtein distance 
to a standardized scale,
typically between 0 and 1, to facilitate comparison across different pairs of strings.
The edit distance is the number of characters that need to be substituted, inserted, 
or deleted, to transform the predicted text into the reference text. The lower the distance, 
the more accurate the model is considered to be.
"""


"""class NormalizedLevenshteinDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.levenshtein_distance = EditDistance(reduction="sum")

    def forward(self, predictions, targets):
        total_chars = sum(len(label) for label in targets)
        total_distance = self.levenshtein_distance(predictions, targets)
        result = (total_chars - total_distance) / total_chars
        return result
"""


class NormalizedLevenshteinDistance(Metric):
    def __init__(self, **kwargs: torch.Any):
        super().__init__()
        text = None

    def update(self):
        pass

    def compute(self, predictions, targets):
        total_chars = sum(len(label) for label in targets)
