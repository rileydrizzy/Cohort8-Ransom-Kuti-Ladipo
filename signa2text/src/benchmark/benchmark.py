"""
Module for benchmarking a PyTorch model.

This module provides a `BenchMarker` class for analyzing model metrics such as
Multiply-Accumulates(MACs), sparsity, the number of parameters, and model size.

Classes:
- BenchMarker: A class for benchmarking a PyTorch model.

Functions:
- get_model_macs: Calculate the MACs (Multiply-Accumulates) of a model.
- get_model_sparsity: Calculate the sparsity of a model.
- get_num_parameters: Calculate the total number of parameters of a model.
- get_model_size: Calculate the size of a model in bits.


"""
from torchprofile import profile_macs
from torch import nn

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


class BenchMarker:
    """
    Benchmarking class to analyze model metrics such as MACs,
    sparsity, number of parameters, and model size.
    """

    def __init__(self) -> None:
        pass

    def get_model_macs(self, model, inputs=None) -> int:
        """
        Calculate the Multiply-Accumulates (MACs) of a model.

        Parameters:
            - model: The PyTorch model.
            - inputs: The input tensor to the model.

        Returns:
            - int: The number of MACs.
        """
        return profile_macs(model, inputs)

    def get_model_sparsity(self, model: nn.Module) -> float:
        """
        Calculate the sparsity of the given model.

        Sparsity is defined as 1 - (number of non-zeros / total number of elements).

        Parameters:
            - model: The PyTorch model.

        Returns:
            - float: The sparsity of the model.
        """
        num_nonzeros, num_elements = 0, 0
        for param in model.parameters():
            num_nonzeros += param.count_nonzero()
            num_elements += param.numel()
        return 1 - float(num_nonzeros) / num_elements

    def get_num_parameters(self, model: nn.Module, count_nonzero_only=False) -> int:
        """
        Calculate the total number of parameters of the model.

        Parameters:
            - model: The PyTorch model.
            - count_nonzero_only: If True, count only nonzero weights.

        Returns:
            - int: The total number of parameters.
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
        Calculate the model size in bits.

        Parameters:
            - model: The PyTorch model.
            - data_width: Number of bits per element.
            - count_nonzero_only: If True, count only nonzero weights.

        Returns:
            - int: The model size in bits.
        """
        return self.get_num_parameters(model, count_nonzero_only) * data_width

    def runner(self, model):
        """
        Run the benchmark on the given model.

        Parameters:
            - model: The PyTorch model.

        Returns:
            - tuple: A tuple containing the model metrics
        """
        model_macs = self.get_model_macs(model)
        model_sparsity = self.get_model_sparsity(model)
        model_num_params = self.get_num_parameters(model)
        model_size = self.get_model_size(model)

        return model_macs, model_sparsity, model_num_params, model_size