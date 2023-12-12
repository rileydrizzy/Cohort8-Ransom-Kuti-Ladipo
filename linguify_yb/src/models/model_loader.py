"""doc

"""

from models.baseline_transformer import ASLTransformer


class ModelLoader:
    """Model Loader"""

    def __init__(self):
        self.models = {
            "asl_transfomer": ASLTransformer(),
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
            raise ValueError("Model is not in the model list")
