"""doc

"""
from linguify_yb.src.models import asl_transfomer


class ModelLoader:
    """_summary_"""

    def __init__(self):
        self.models = {"asl_transfomer": asl_transfomer.build_model()}

    def get_model(self, model_name):
        """_summary_

        Parameters
        ----------
        model_name : str
            _description_

        Returns
        -------
        object
            returns model
        """

        if model_name in self.models:
            return self.models[model_name]
        else:
            return False
