"""
ASL Fingerspelling Frame Features Module

This module defines constants and lists related to ASL Fingerspelling frame features.

Variables:
- FRAME_LEN: Length of each frame.
- LIP: Indices corresponding to lip features.
- FEATURE_COLUMNS: Combined list of feature columns, including frame, hand, pose, and face features.
"""

# Length of each frame
FRAME_LEN = 128

# Indices corresponding to lip features
LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321,
    375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318,
    324, 308,
]

# Feature names for different body parts
FRAME = ["frame"]
N_LHAND = (
    [f"x_left_hand_{i}" for i in range(21)]
    + [f"y_left_hand_{i}" for i in range(21)]
    + [f"z_left_hand_{i}" for i in range(21)]
)

N_RHAND = (
    [f"x_right_hand_{i}" for i in range(21)]
    + [f"y_right_hand_{i}" for i in range(21)]
    + [f"z_right_hand_{i}" for i in range(21)]
)
N_POSE = (
    [f"x_pose_{i}" for i in range(33)]
    + [f"y_pose_{i}" for i in range(33)]
    + [f"z_pose_{i}" for i in range(33)]
)
N_FACE = (
    [f"x_face_{i}" for i in LIP]
    + [f"y_face_{i}" for i in LIP]
    + [f"z_face_{i}" for i in LIP]
)

# Combined list of feature columns
FEATURE_COLUMNS = FRAME + N_LHAND + N_RHAND + N_POSE + N_FACE
