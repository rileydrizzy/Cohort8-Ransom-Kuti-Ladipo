import torch.nn as nn


class TestLinear(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)
        self.sequnn = nn.Sequential(self.linear1, self.linear2, self.linear3)

    def forward(self, input_x):
        outs = self.linear1(input_x)
        return outs


def build_model():
    return TestLinear()
