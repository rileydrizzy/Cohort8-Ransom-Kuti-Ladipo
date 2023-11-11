"""doc
"""
import torch

from linguify_yb.src.dataset.frames_config import (FRAME_LEN, LHAND_IDX,
                                                   LPOSE_IDX, RHAND_IDX,
                                                   RPOSE_IDX)

# TODO Clean up code, add comments and docs
# TODO remove print and debug statements

# Preprocess frame


def resize_pad(x):
    if x.shape[0] < FRAME_LEN:
        x = torch.nn.functional.pad(x, (0, 0, 0, FRAME_LEN - x.shape[0], 0, 0))
    else:
        x = x.unsqueeze(0)  # Add batch and channel dimensions
        x = torch.nn.functional.interpolate(
            x, size=(FRAME_LEN, x.shape[1]), mode="bilinear", align_corners=False
        ).squeeze(0)

    return x


def frames_preprocess(x):
    x = torch.tensor(x)
    rhand = x[:, RHAND_IDX]
    lhand = x[:, LHAND_IDX]
    rpose = x[:, RPOSE_IDX]
    lpose = x[:, LPOSE_IDX]

    rnan_idx = torch.any(torch.isnan(rhand), dim=1)
    lnan_idx = torch.any(torch.isnan(lhand), dim=1)

    rnans = torch.sum(rnan_idx)
    lnans = torch.sum(lnan_idx)

    if rnans > lnans:
        hand = lhand
        pose = lpose

        hand_x = hand[:, 0 * (len(LHAND_IDX) // 3) : 1 * (len(LHAND_IDX) // 3)]
        hand_y = hand[:, 1 * (len(LHAND_IDX) // 3) : 2 * (len(LHAND_IDX) // 3)]
        hand_z = hand[:, 2 * (len(LHAND_IDX) // 3) : 3 * (len(LHAND_IDX) // 3)]
        hand = torch.cat([1 - hand_x, hand_y, hand_z], dim=1)

        pose_x = pose[:, 0 * (len(LPOSE_IDX) // 3) : 1 * (len(LPOSE_IDX) // 3)]
        pose_y = pose[:, 1 * (len(LPOSE_IDX) // 3) : 2 * (len(LPOSE_IDX) // 3)]
        pose_z = pose[:, 2 * (len(LPOSE_IDX) // 3) : 3 * (len(LPOSE_IDX) // 3)]
        pose = torch.cat([1 - pose_x, pose_y, pose_z], dim=1)
    else:
        hand = rhand
        pose = rpose

    hand_x = hand[:, 0 * (len(LHAND_IDX) // 3) : 1 * (len(LHAND_IDX) // 3)]
    hand_y = hand[:, 1 * (len(LHAND_IDX) // 3) : 2 * (len(LHAND_IDX) // 3)]
    hand_z = hand[:, 2 * (len(LHAND_IDX) // 3) : 3 * (len(LHAND_IDX) // 3)]
    hand = torch.cat(
        [hand_x.unsqueeze(-1), hand_y.unsqueeze(-1), hand_z.unsqueeze(-1)], dim=-1
    )

    mean = torch.mean(hand, dim=1).unsqueeze(1)
    std = torch.std(hand, dim=1).unsqueeze(1)
    hand = (hand - mean) / std

    pose_x = pose[:, 0 * (len(LPOSE_IDX) // 3) : 1 * (len(LPOSE_IDX) // 3)]
    pose_y = pose[:, 1 * (len(LPOSE_IDX) // 3) : 2 * (len(LPOSE_IDX) // 3)]
    pose_z = pose[:, 2 * (len(LPOSE_IDX) // 3) : 3 * (len(LPOSE_IDX) // 3)]
    pose = torch.cat(
        [pose_x.unsqueeze(-1), pose_y.unsqueeze(-1), pose_z.unsqueeze(-1)], dim=-1
    )

    x = torch.cat([hand, pose], dim=1)
    print(f"befor  re{x.shape}")
    x = resize_pad(x)
    print(f"after re{x.shape}")
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    print(x.shape)

    #! CRITICAL Debug
    # x = x.view(FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX))
    return x
