import json
from copy import deepcopy

import torch
import torch.nn.functional as F

import editdistance

from transformers import AutoTokenizer


def word_edit_distance(source, target):
    source = source.split(" ")
    target = target.split(" ")
    return editdistance.eval(source, target), len(target)


def word_error_count(preds_0, targets_0):
    preds = deepcopy(preds_0)
    targets = deepcopy(targets_0)
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(targets, list):
        targets = [targets]
    assert len(preds) == len(targets), "Lengths of prediction and target lists don't match"
    errors = 0
    words  = 0
    for pred, target in zip(preds, targets):
        new_errors, new_words = word_edit_distance(pred,target)
        errors += new_errors
        words += new_words
    return errors, words    


def format_ctc(pred, vocab, blank_id):
    phonemes = []
    last = -1
    for idx in pred:
        if idx != last and idx != blank_id:
            phonemes.append(vocab[idx])
            last = deepcopy(idx)
    return phonemes


def per(model, model_inputs, unused_inputs, outputs, config, **kwargs):
    blank_id = config.method.model_kwargs.blank_id
    vocab = json.load(open(config.data.vocab_file,"r"))
    preds = outputs["preds"].argmax(-1)
    preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
    phonemes = [" ".join(p) for p in unused_inputs["phonemes"]]
    errors, n_phonemes = word_error_count(preds, phonemes)
    for i in range(kwargs["n_print"]):
        print(
            "\n-----\n ", 
            preds[i].replace(" ","").replace("SIL"," SIL "), 
            "\n-----\n ", 
            phonemes[i].replace(" ","").replace("SIL"," SIL "), 
            "\n-----\n ", 
            unused_inputs["sentence"][i], 
            "\n-----\n\n "
        )
    return torch.tensor(errors/n_phonemes, device=model_inputs["spikes"].device)


def wer(model, model_inputs, unused_inputs, outputs, config, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path, add_bos_token=False, add_eos_token=False)
    preds = outputs["preds"].argmax(-1)[:,:-1]
    targets = outputs["targets"][:,1:]
    pred_sentences = [tokenizer.decode(p[t!=-100], skip_special_tokens=True) for t, p in zip(targets, preds)]
    target_sentences = unused_inputs["sentence"]
    errors, n_words = word_error_count(pred_sentences, target_sentences)
    print(pred_sentences, "\n-----\n")
    print(target_sentences, "\n-----\n\n ")
    return torch.tensor(errors/n_words, device=model_inputs["spikes"].device)


def eval_wer(model, model_inputs, unused_inputs, outputs, config, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path, add_bos_token=False, add_eos_token=False)
    unk_token_id = tokenizer.vocab_size
    prompt_ids = model_inputs["input_ids"][
        torch.logical_and(model_inputs["targets"] == -100, model_inputs["input_ids"] != unk_token_id)
    ]

    if len(prompt_ids.size()) == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    attention_mask = torch.ones_like(prompt_ids)

    model_inputs.update({
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
    })
    model_inputs.pop("targets")
    
    beams = kwargs["n_beams"]
    if beams > 1:
        gen_config = {
            "max_new_tokens": 20, 
            "do_sample": False, 
            "num_beams": beams, 
            "num_beam_groups": 1, 
            "diversity_penalty": 0.0,
            "repetition_penalty": 1.0, 
            "length_penalty": 1.0, 
            "renormalize_logits": True, 
            "low_memory": True,
            "num_return_sequences": beams, "output_scores": True, "return_dict_in_generate": True,
            "pad_token_id": tokenizer.unk_token_id,
        }
    else:
        gen_config = {
            "max_new_tokens": 20, 
            "do_sample": False,
            "low_memory": True,
            "pad_token_id": tokenizer.unk_token_id,
        }

    pred = model.generate(**model_inputs, **gen_config)
    if beams > 1:
        pred = pred.sequences[0]
    else:
        pred = pred[0]
    
    pred_sentence = tokenizer.decode(pred, skip_special_tokens=True).strip()
    target_sentence = unused_inputs["sentence"][0]
    errors, n_words = word_error_count(pred_sentence, target_sentence)
    print(pred_sentence, "\n-----\n")
    print(target_sentence, "\n-----\n\n ")
    return torch.tensor(errors/n_words)