"""doc
"""
import torch
from torch.nn import functional as F

# from dataset.frames_config import FRAME_LEN

# TODO Clean up code, add comments and docs
# TODO remove print and debug statements


def clean_frames_process(
    x, max_frame_len=128, n_hand_landmarks=21, n_pose_landmarks=33, n_face_landmarks=40
):
    x = x[:max_frame_len]
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    n_frames = x.size(0)
    lhand = x[:, 0:63].view(n_frames, 3, n_hand_landmarks).transpose(1, 2)
    rhand = x[:, 63:126].view(n_frames, 3, n_hand_landmarks).transpose(1, 2)
    pose = x[:, 126:225].view(n_frames, 3, n_pose_landmarks).transpose(1, 2)
    face = x[:, 225:345].view(n_frames, 3, n_face_landmarks).transpose(1, 2)

    x = torch.cat([lhand, rhand, pose, face], axis=1)
    x = x.view(n_frames, 345)
    if n_frames < max_frame_len:
        # Calculate the padding on the first dimension from the bottom
        padding_bottom = max(0, max_frame_len - x.size(0))
        # Pad the tensor along the first dimension from the bottom
        x = F.pad(x, (0, 0, 0, padding_bottom))
    return x
