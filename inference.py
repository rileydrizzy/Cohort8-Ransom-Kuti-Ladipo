"""doc
"""
import os
import cv2

import numpy as np
import pandas as pd
import mediapipe as mp
import torch
from IPython.display import Audio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel
from linguify_yb.src.models import baseline_transfomer

nllb_model_name = "facebook/nllb-200-distilled-600M"
mms_model_name = "facebook/mms-tts-yor"
youruba_lang = "yor_Latn"


# NLLB Model
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

# MMS Model
mms_model = VitsModel.from_pretrained(mms_model_name)
mms_tokenizer = AutoTokenizer.from_pretrained(mms_model_name)


def NLLB_infer(eng_text):
    inputs = nllb_tokenizer(eng_text, return_tensors="pt")
    translate_token = nllb_model.generate(
        **inputs,
        forced_bos_token_id=nllb_tokenizer.lang_code_to_id[youruba_lang],
        max_length=50
    )
    outputs = nllb_tokenizer.batch_decode(translate_token, skip_special_tokens=True)[0]
    return outputs


def MMS_model_infer(text):
    inputs = mms_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = mms_model(**inputs).waveform
    return Audio(output, rate=mms_model.config.sampling_rate)


# TODO Debug
def extract_landmarks(path, start_frame=0):
    mp_holistic = mp.solutions.holistic
    # Initialize variables
    frame_number = 0
    frame = []
    type_ = []
    index = []
    x = []
    y = []
    z = []

    # Open the video file
    cap = cv2.VideoCapture(path)

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
