"""doc
"""

FRAME_LEN = 128

LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = (
    [f"x_right_hand_{i}" for i in range(21)]
    + [f"x_left_hand_{i}" for i in range(21)]
    + [f"x_pose_{i}" for i in POSE]
)
Y = (
    [f"y_right_hand_{i}" for i in range(21)]
    + [f"y_left_hand_{i}" for i in range(21)]
    + [f"y_pose_{i}" for i in POSE]
)
Z = (
    [f"z_right_hand_{i}" for i in range(21)]
    + [f"z_left_hand_{i}" for i in range(21)]
    + [f"z_pose_{i}" for i in POSE]
)

FEATURE_COLUMNS = X + Y + Z

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [
    i
    for i, col in enumerate(FEATURE_COLUMNS)
    if "pose" in col and int(col[-2:]) in RPOSE
]
LPOSE_IDX = [
    i
    for i, col in enumerate(FEATURE_COLUMNS)
    if "pose" in col and int(col[-2:]) in LPOSE
]
