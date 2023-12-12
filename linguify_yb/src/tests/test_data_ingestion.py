"doc"

import pytest

import torch
from torch.utils.data import DataLoader
from src.dataset.frames_config import FRAME_LEN
from src.dataset.preprocess import clean_frames_process
from src.dataset.dataset_loader import TokenHashTable

# TODO test for frames in right shapes, in tensor, frames are normalize
# TODO test for frames dont contain NAN

# TODO test for labels are tokensize


@pytest.mark.parametrize(
    "frames",
    [torch.randn(num_frames, 345) for num_frames in [10, 108, 128, 156, 750, 420]],
)
def test_frames_preprocess(frames):
    """doc"""
    frames = clean_frames_process(frames)
    expected_output_shape = (128, 345)
    assert (
        expected_output_shape == frames.shape
    ), f"frames shape should be {expected_output_shape}"


def test_token_hash_table():
    token_table = TokenHashTable()
    sample_sentence = "this is a test run"
    sample_sentence_len = len(sample_sentence)
    sample_sentence_token = [60,51,39,40,50,0,40,50,0,32,0,51,36,50,51,0,49,52,45,61]
    # Padding the token
    sample_sentence_token = sample_sentence_token + (
        [59] * (64 - len(sample_sentence_token))
    )
    sample_sentence_token = torch.tensor(sample_sentence_token)
    tokenize_result = token_table.sentence_to_tensor(sample_sentence)
    assert sample_sentence_len == len(tokenize_result)
    assert sample_sentence_token == tokenize_result

    # Assert that clean_frames is a PyTorch tensor
    assert torch.is_tensor(tokenize_result), "is not PyTorch tensor"
