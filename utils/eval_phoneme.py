import re
import pandas as pd
import os
import torch
import numpy as np
import time

import editdistance
import inspect


def rearrange_speech_logits_pt(logits):
    # original order is [BLANK, phonemes..., SIL]
    # rearrange so the order is [BLANK, SIL, phonemes...]
    logits = np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)
    return logits

def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)

def compute_wer(lm_results):
    total_true_length = 0
    total_edit_distance = 0
    lm_results["edit_distance"] = []
    lm_results["num_words"] = []

    for i in range(len(lm_results["pred_sentence"])):
        true = remove_punctuation(lm_results["true_sentence"][i]).strip()
        pred = remove_punctuation(lm_results["pred_sentence"][i]).strip()
        ed = editdistance.eval(true.split(), pred.split())
        total_true_length += len(true.split())
        total_edit_distance += ed
        lm_results["edit_distance"].append(ed)
        lm_results["num_words"].append(len(true.split()))

        print(f"True sentence:       {true}")
        print(f"Predicted sentence:  {pred}")
        print(f"WER: {ed} / {100 * len(true.split())} = {ed / len(true.split()):.2f}%\n")

    print(f"Total true sentence length: {total_true_length}")
    print(f"Total edit distance: {total_edit_distance}")
    print(f"Aggregate WER: {100 * total_edit_distance / total_true_length:.2f}%")

def save_results(lm_results, model_path, eval_type):
    output_file = os.path.join(model_path, f"baseline_{eval_type}_predicted_sentences_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    df = pd.DataFrame({"id": list(range(len(lm_results["pred_sentence"]))), "text": lm_results["pred_sentence"]})
    df.to_csv(output_file, index=False)

def setup_device(gpu_number):
    if torch.cuda.is_available() and gpu_number >= 0:
        if gpu_number >= torch.cuda.device_count():
            raise ValueError(f"GPU number {gpu_number} is out of range.")
        device = torch.device(f"cuda:{gpu_number}")
        print(f"Using {device} for model inference.")
    else:
        if gpu_number >= 0:
            print(f"GPU number {gpu_number} requested but not available.")
        device = torch.device("cpu")
        print("Using CPU for model inference.")
    return device

def get_model_inputs(model):
    model_to_inspect = model
    # Inspect the base model in case a peft adapter is loaded
    if getattr(model, "peft_type", None) is not None:
        if hasattr(model, "get_base_model"):
            model_to_inspect = model.get_base_model()
        else:
            model_to_inspect = model.base_model.model
    signature = inspect.signature(model_to_inspect.forward)
    model_inputs = list(signature.parameters.keys())
    return model_inputs

