"""doc

"""
import torch.nn as nn
from linguify_yb.src.models import baseline_transfomer, test_model

class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {
            "asl_transfomer": baseline_transfomer.build_model(),
            "test_model": test_model.build_model(),
        }

    def get_model(self, model_name):
        """build and retrieve the model instance

        Parameters
        ----------
        model_name : str
            model name

        Returns
        -------
        object
            return built model instance
        """

        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError



# For Debugging
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
