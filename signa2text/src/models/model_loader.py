"""doc

"""

from models.baseline_transformer import ASLTransformer
import torch


def test_model():
    model = torch.nn.Sequential(
        [torch.nn.Linear(20, 100), torch.nn.Linear(100, 10), torch.nn.Linear(10, 5)]
    )
    return model


class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {"asl_transfomer": ASLTransformer(), "test_model": test_model()}

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
            raise ValueError("Model is not in the model list")
