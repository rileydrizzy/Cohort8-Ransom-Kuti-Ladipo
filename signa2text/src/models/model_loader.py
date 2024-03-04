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


class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {
            "asl_baseline_transformer": ASLTransformer(),
        }

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
