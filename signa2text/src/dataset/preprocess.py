"""
Module to define a function for cleaning and processing ASL Fingerspelling frames.
Functions:
- clean_frames_process: 

"""

import torch
from torch.nn import functional as F


def clean_frames_process(
    frames,
    max_frame_len=128,
    n_hand_landmarks=21,
    n_pose_landmarks=33,
    n_face_landmarks=40,
):
    """Clean and process ASL Fingerspelling frames.

    Parameters
    ----------
    frames : (torch.Tensor)
        Input tensor containing frames.
    max_frame_len : int, optional
         Maximum length of frames, by default 128
    n_hand_landmarks : int, optional
        Number of hand landmarks, by default 21
    n_pose_landmarks : int, optional
        Number of pose landmarks, by default 33
    n_face_landmarks : int, optional
        Number of face landmarks, by default 40

    Returns
    -------
    frames
       torch.Tensor: Cleaned and processed frames tensor.
    """
    # Clip frames to the maximum length
    frames = frames[:max_frame_len]
    # Replace NaN values with zeros
    frames = torch.where(torch.isnan(frames), torch.zeros_like(frames), frames)

    # Split the tensor into different body part landmarks
    lhand = frames[:, 0:63].view(frames.size(0), 3, n_hand_landmarks).transpose(1, 2)
    rhand = frames[:, 63:126].view(frames.size(0), 3, n_hand_landmarks).transpose(1, 2)
    pose = frames[:, 126:225].view(frames.size(0), 3, n_pose_landmarks).transpose(1, 2)
    face = frames[:, 225:345].view(frames.size(0), 3, n_face_landmarks).transpose(1, 2)

    # Concatenate the landmarks along the specified axis
    frames = torch.cat([lhand, rhand, pose, face], axis=1)
    # Reshape the tensor
    frames = frames.view(frames.size(0), 345)

    if frames.size(0) < max_frame_len:
        # Calculate the padding on the first dimension from the bottom
        padding_bottom = max(0, max_frame_len - frames.size(0))
        # Pad the tensor along the first dimension from the bottom
        frames = F.pad(frames, (0, 0, 0, padding_bottom))

    return frames
