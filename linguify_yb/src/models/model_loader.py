"""doc

"""
from linguify_yb.src.models import asl_transfomer, test_model


class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {
            "asl_transfomer": asl_transfomer.build_model(),
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
