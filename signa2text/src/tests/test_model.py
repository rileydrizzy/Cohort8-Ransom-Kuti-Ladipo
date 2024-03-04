"""doc
"""

import pytest

import torch
from torch.utils.data import DataLoader
from models.baseline_transformer import ASLTransformer


@pytest.fixture
def baseline_model():
    """_summary_"""
    model = ASLTransformer()
    return model


@pytest.mark.parametrize(
    "inputs_x, target_y", [(torch.randn(128, 345), torch.randint(0, 60, (64,)))]
)
def test_baseline_transformer_output_shape(baseline_model, inputs_x, target_y):
    """_summary_"""
    output = baseline_model(inputs_x, target_y)
    # Assert
    expected_output_shape = (64, 62)
    assert output.shape == expected_output_shape


@pytest.mark.parametrize("inputs_x", [(torch.randn(128, 345)), (torch.randn(128, 345))])
def test_baseline_transformer_generate_out(
    baseline_model,
    inputs_x,
):
    """_summary_"""
    output = baseline_model.generate(inputs_x)
    # Assert
    expected_output_len = 64
    assert len(output) == expected_output_len


@pytest.mark.parametrize(
    "inputs_x, target_y, batch_size",
    [
        (
            torch.randn(batch_size, 128, 345),
            torch.randint(0, 60, (batch_size, 64)),
            batch_size,
        )
        for batch_size in [1, 2, 4, 8]
    ],
)
def test_baseline_transformer_batch_shape(
    baseline_model, inputs_x, target_y, batch_size
):
    output = baseline_model(inputs_x, target_y)
    expected_output_shape = (batch_size, 64, 62)
    assert output.shape == expected_output_shape
