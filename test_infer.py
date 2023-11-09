"""doc
"""
import os

import cv2
import mediapipe as mp
import torch
from IPython.display import Audio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

from linguify_yb.src.models import asl_transfomer

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
