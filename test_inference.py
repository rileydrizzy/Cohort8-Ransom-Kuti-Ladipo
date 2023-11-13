"""doc
"""
import os
import cv2
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
def extract_pose_frames(video_file):
    # Initialize Mediapipe holistic model
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    # Open a video file
    video_path = video_file
    cap = cv2.VideoCapture(video_path)

    # Loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = holistic.process(rgb_frame)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
            # mp.solutions.drawing_utils.draw_landmarks(
            #   frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
        # TODO BUG!
        # Display the frame
        cv2.imshow("Holistic Landmarks", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
