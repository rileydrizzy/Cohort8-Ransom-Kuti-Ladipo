"""
This module provides utility functions for the NSL-2-AUDIO  web application.

Functions:
- `load_lottiefile(filepath: str) -> dict`: Loads a Lottie animation file (.json) \
    and returns its parsed contents as a dictionary.
- `extract_landmarks_features(video_file, startframe)`: Extracts landmarks features \
    from the uploaded NSL video file.


Dependencies:
- cv2
- mediapipe
- pandas
- json

Usage:
Import this module and use the provided functions as needed.

"""

import json
import cv2
import pandas as pd
import mediapipe as mp


def load_lottiefile(filepath: str) -> dict:
    """Loads a Lottie animation file (.json) and returns its parsed contents as a dictionary.

    Parameters
    ----------
    filepath : str
        Path to the Lottie animation file.

    Returns
    -------
    dict
        The parsed contents of the Lottie animation file.

    Raises
    -------
    FileNotFoundError
        If the specified file is not found.
    """

    with open(filepath, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Lottie animation file not found: {filepath}"
            ) from error


# TODO Debug
def extract_landmarks_features(cap, start_frame=0):

    if not isinstance(cap, cv2.VideoCapture):
        raise ValueError

    #
    mp_holistic = mp.solutions.holistic
    # Initialize variables
    frame_number = 0
    frame = []
    type_ = []
    index = []
    x = []
    y = []
    z = []

    # Get the total number of frames in the video
    end_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frames per second (FPS) of the video
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # cap.set(cv2.CAP_PROP_FPS, fps)

    # Initialize holistic model for landmark detection
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            success, image = cap.read()

            # Break if video is finished
            if not success:
                break

            # Increment frame number
            frame_number += 1

            # Skip frames until the start_frame is reached
            if frame_number < start_frame:
                continue

            # Break if end_frame is reached
            if end_frames != -1 and frame_number > end_frames:
                break

            # Prepare image for landmark detection
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Process face landmarks
            if results.face_landmarks is None:
                for i in range(478):
                    frame.append(frame_number)
                    type_.append("face")
                    index.append(i)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind, val in enumerate(results.face_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("face")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)

            # Process pose landmarks
            if results.pose_landmarks is None:
                for i in range(32):
                    frame.append(frame_number)
                    type_.append("pose")
                    index.append(i)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind, val in enumerate(results.pose_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("pose")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)

            # Process left hand landmarks
            if results.left_hand_landmarks is None:
                for i in range(20):
                    frame.append(frame_number)
                    type_.append("left_hand")
                    index.append(i)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind, val in enumerate(results.left_hand_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("left_hand")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)

            # Process right hand landmarks
            if results.right_hand_landmarks is None:
                for i in range(20):
                    frame.append(frame_number)
                    type_.append("right_hand")
                    index.append(i)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind, val in enumerate(results.right_hand_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("right_hand")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)
    # TODO rearrange dataframe to account for just the frames in sequential manner
    # TODO consider to use numpy instead of a dataframe
    # Create a DataFrame from the collected data
    return pd.DataFrame(
        {"frame": frame, "type": type_, "landmark_index": index, "x": x, "y": y, "z": z}
    )
