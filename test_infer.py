"""doc
"""
import os
import wandb
import torch
from transformers import VitsModel, AutoTokenizer, AutoModelForSeq2SeqLM
from linguify_yb.models import asl_transfomer

nllb_model_name = 'facebook/nllb-200-distilled-600M'
mms_model_name = "facebook/mms-tts-yor"
youruba_lang = 'yor_Latn'

mms_model = VitsModel.from_pretrained(mms_model_name)
mms_tokenizer = AutoTokenizer.from_pretrained(mms_model_name)
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
