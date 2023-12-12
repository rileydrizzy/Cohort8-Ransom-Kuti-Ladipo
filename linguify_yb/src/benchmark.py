"""doc
"""
from torchprofile import profile_macs
from torch import nn


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


class BenchMarker:
    """_summary_"""

    def __init__(self) -> None:
        pass

    def get_model_macs(self, model, inputs=None) -> int:
        """
        calculate the MACS of a model
        """
        return profile_macs(model, inputs)

    def get_model_sparsity(self, model: nn.Module) -> float:
        """
        calculate the sparsity of the given model
            sparsity = #zeros / #elements = 1 - #nonzeros / #elements
        """
        num_nonzeros, num_elements = 0, 0
        for param in model.parameters():
            num_nonzeros += param.count_nonzero()
            num_elements += param.numel()
        return 1 - float(num_nonzeros) / num_elements

    def get_num_parameters(self, model: nn.Module, count_nonzero_only=False) -> int:
        """
        calculate the total number of parameters of model
        :param count_nonzero_only: only count nonzero weights
        """
        num_counted_elements = 0
        for param in model.parameters():
            if count_nonzero_only:
                num_counted_elements += param.count_nonzero()
            else:
                num_counted_elements += param.numel()
        return num_counted_elements

    def get_model_size(
        self, model: nn.Module, data_width=32, count_nonzero_only=False
    ) -> int:
        """
        calculate the model size in bits
        :param data_width: #bits per element
        :param count_nonzero_only: only count nonzero weights
        """
        return self.get_num_parameters(model, count_nonzero_only) * data_width

    def runner(self, model):
        model_macs = self.get_model_macs(model)
        model_sparsity = self.get_model_sparsity(model)
        model_num_params = self.get_num_parameters(model)
        model_size = self.get_model_size(model)

        return
