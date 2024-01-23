"""
Test Module for Dataset Ingestion and Preprocessing

This module contains test functions for dataset ingestion, preprocessing and tokenization.

Functions:
- test_frames_preprocess(frames): Tests the preprocessing of frames.

- test_token_hash_table(): Tests the TokenHashTable for sentence tokenization.
"""

import pytest

import torch
from dataset.frames_config import FRAME_LEN
from dataset.preprocess import preprocess_frames
from dataset.dataset_loader import TokenHashTable


@pytest.mark.parametrize(
    "frames",
    [torch.randn(num_frames, 345) for num_frames in [10, 108, 128, 156, 750, 420]],
)
def test_frames_preprocess(frames):
    """Tests the preprocessing of frames.

    Parameters
    ----------
    frames : torch.Tensor
        Input tensor containing frames for preprocessing.
    """
    frames = preprocess_frames(frames)
    expected_output_shape = (128, 345)
    assert (
        expected_output_shape == frames.shape
    ), f"Frames shape should be {expected_output_shape}, got {frames.shape}"


def test_token_hash_table():
    """
    Tests the TokenHashTable for sentence tokenization.
    """
    token_lookup_table = TokenHashTable()
    sample_sentence = "this is a test run"
    sample_sentence_len = len(sample_sentence)
    sample_sentence_tokens = [
        51,
        39,
        40,
        50,
        0,
        40,
        50,
        0,
        32,
        0,
        51,
        36,
        50,
        51,
        0,
        49,
        52,
        45,
    ]
    sample_sentence_tokens = torch.tensor(sample_sentence_tokens, dtype=torch.long)
    tokenize_result = token_lookup_table.sentence_to_tensor(sample_sentence)

    # Assert the length of tokenize text
    assert sample_sentence_len == len(
        tokenize_result
    ), f"Expexted length of tokenize text to be {sample_sentence_len}, got {len(tokenize_result)}"

    is_same = all(
        torch.equal(idx1, idx2)
        for idx1, idx2 in zip(sample_sentence_tokens, tokenize_result)
    )
    # Assert tokens match the expected value
    assert is_same == True, "Tokens do not match the expected value"

    # Assert that clean_frames is a PyTorch tensor
    assert torch.is_tensor(tokenize_result), "Tokens are not PyTorch tensor"
