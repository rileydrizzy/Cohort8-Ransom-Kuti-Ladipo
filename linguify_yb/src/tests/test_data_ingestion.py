"doc"

import pytest

import torch
from torch.utils.data import DataLoader
from src.dataset.frames_config import FRAME_LEN
from src.dataset.preprocess import clean_frames_process


@pytest.mark.parametrize(
    "frames",
    [torch.randn(num_frames, 345) for num_frames in [10, 108, 128, 156, 750, 420]],
)
def test_frames_preprocess(frames):
    clean_frames = clean_frames_process(frames)
    expected_output_shape = (128, 345)
    assert expected_output_shape == clean_frames.shape
