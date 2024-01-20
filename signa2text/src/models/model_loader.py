"""
Model Loader Module

This module defines a class for loading different models.

Classes:
- ModelLoader: Loads various models.

Methods:
- get_model(model_name): Builds and retrieves a specific model instance.

"""
import torch
from models.baseline_transformer import ASLTransformer

# TODO remove test model
def test_model():
    model = torch.nn.Sequential(
        [torch.nn.Linear(20, 100), torch.nn.Linear(100, 10), torch.nn.Linear(10, 5)]
    )
    return model


class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {"asl_transformer": ASLTransformer(), "test_model": test_model()}

    def get_model(self, model_name):
        """Build and retrieve the model instance.

        Parameters
        ----------
        model_name : str
            Name of the model.

        Returns
        -------
        object
            Built model instance.
        """
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError("Model is not in the model list")
