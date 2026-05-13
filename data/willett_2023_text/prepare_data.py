import os
import re
import json
import string
from glob import glob
from tqdm import tqdm
from typing import List, Optional, Any, Dict, Tuple

from collections import defaultdict

import scipy
import numpy as np

from transformers import PreTrainedTokenizer

# Time bin size: 20 ms

def load_data(
    data_dir:        str, 
    day_idxs:        Optional[List[int]] = None,
    zscore_block:    Optional[bool] = False, 
    zscore_day:      Optional[bool] = False,
    features:        Optional[List[str]] = ["tx1"], 
    area_start:      Optional[int] = 0, 
    area_end:        Optional[int] = 256,
    **kwargs,
) -> Dict[str, List[Dict[str, Any]]]:
    
    def load_single_file(file: str, zscore_block: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        data = scipy.io.loadmat(file)
        y_i = data["sentenceText"]
        b_i = data["blockIdx"]
        d_i = tuple(file.split("/")[-1].split(".")[1:4])

        x_i = np.array(
            [
                np.concatenate([data[feature][0,i][:, area_start:area_end] for feature in features], axis=1) 
                for i in range(len(y_i))
            ], dtype = np.ndarray
        )
        if zscore_block:
            x_i = zscore_by_block(x_i, b_i)
            
        return (
            x_i, y_i, b_i, [d_i] * len(y_i)
        )

    def zscore_by_block(x_i: np.ndarray, b_i: np.ndarray) -> np.ndarray:
        blocks = set([block for [block] in b_i.tolist()])
        for block in blocks:
            idx = np.where(b_i == block)[0]
            mu = np.mean(np.concatenate(x_i[idx], axis=0), axis=0)
            sd = np.std(np.concatenate(x_i[idx], axis=0), axis=0)
            for i in idx:
                x_i[i] = (x_i[i] - mu) / sd
        return x_i

    def get_split_dict(split_dir: str, zscore_block: bool) -> List[Dict]:
        all_files = sorted(
            glob(os.path.join(split_dir, "*")), 
            key=lambda file: tuple(file.split("/")[-1].split(".")[1:4])
        )
        
        x, y, b, d = [], [], [], []
        for file in tqdm(all_files):
            x_i, y_i, b_i, d_i = load_single_file(file, zscore_block)
            x.append(x_i)
            y.append(y_i)
            b.append(b_i)
            d.extend(d_i)

        x = np.concatenate(x).tolist()
        y = np.concatenate(y)
        b = (np.concatenate(b).squeeze() - 1).tolist()  # translate to start at 0

        return [
            {
                "spikes": x_i.astype(np.float32),
                "sentence": y_i.translate(str.maketrans("", "", string.punctuation.replace("'",""))).lower().strip(),
                "block": b_i,
                "day": d_i,
            } for x_i, y_i, b_i, d_i in zip(x, y, b, d)
        ]

    def zscore_by_day(dataset_dict: Dict, day_idxs: List[int]) -> Dict:
        spikes_by_day = {
            i: np.concatenate(
                [row["spikes"] for row in dataset_dict["train"] if int(row["day_idx"]) == i], axis=0
            ) for i in day_idxs
        }
        spikes_mean = {i: np.mean(v, axis=0) for i, v in spikes_by_day.items()}
        spikes_std = {i: np.std(v, axis=0) for i, v in spikes_by_day.items()}

        for split in dataset_dict:
            for i, row in enumerate(dataset_dict[split]):
                dataset_dict[split][i]["spikes"] = (
                    (dataset_dict[split][i]["spikes"] - spikes_mean[int(row["day_idx"])]) / spikes_std[int(row["day_idx"])]
                )
        return dataset_dict

    splits = {"train": "train", "val": "test", "holdout": "competitionHoldOut"}
    dataset_dict = {
        k: get_split_dict(os.path.join(data_dir, v), zscore_block) for k, v in splits.items()
    }

    all_blocks = set([row["block"] for split in splits for row in dataset_dict[split]])
    all_days = sorted(set([row["day"] for split in splits for row in dataset_dict[split]]))
    day_idxs = day_idxs if day_idxs is not None else list(range(len(all_days)))

    d_to_i = {d: i for i, d in enumerate(all_days)}
    b_to_i = {b: i for i, b in enumerate(all_blocks)}

    for split in splits:
        dataset_dict[split] = [
            {
                **row, 
                "block_idx": np.asarray(b_to_i[row["block"]]),
                "day_idx": np.asarray(d_to_i[row["day"]])
            }
            for row in dataset_dict[split] if d_to_i[row["day"]] in day_idxs
        ]

    dataset_dict = split_val(dataset_dict)

    if zscore_day:
        dataset_dict = zscore_by_day(dataset_dict, day_idxs)

    return dataset_dict


def split_val(dataset_dict: Dict, seed: int = 42) -> Dict:
    val_data = dataset_dict.get("val", [])
    n = len(val_data)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    
    half = n // 2
    val_indices = perm[:half]
    test_indices = perm[half:]

    dataset_dict["val"] = [val_data[i] for i in val_indices]
    dataset_dict["test"] = [val_data[i] for i in test_indices]

    return dataset_dict


def uni2bigram(phonelist, phone_len, insert_selfloop=True, C=40):
    bigramlist = []
    for i, label in enumerate(phonelist[:phone_len]):          
        if i == 0:
            bigramlist.append(C*(C-1) + label)                 # START -> label
        else:
            bigramlist.append((phonelist[i-1]-1)*C + label)    # prev -> curr
        if insert_selfloop:
            bigramlist.append((label-1)*C + label)             # self-loop
    new_phone_len = len(bigramlist)
    return bigramlist, new_phone_len


def create_phonemes_ctc_labels(
    dataset:    Dict[str,List[Dict[str,Any]]], 
    vocab_file: str,
) -> Dict[str,List[Dict[str,Any]]]:
    
    from g2p_en import G2p
    g2p = G2p() # graphme to phoneme
    vocab = json.load(open(vocab_file,"r"))

    # sentence to phonemes
    def s_to_p(s: str) -> List[str]:
        # keep only phonemes and add SIL at the end so that every word ends in SIL
        return [re.sub(r'[0-9]','',pp) if pp != " " else "SIL" for pp in g2p(s) if re.match(r'[A-Z]+', pp) or pp == " "] + ["SIL"] 

    # phonemes to vocab index
    def p_to_i(p: List[str]) -> List[int]:
        return [vocab.index(pp) for pp in p]

    for split in dataset:
        for i, row in enumerate(dataset[split]):
            phonemes = s_to_p(row["sentence"])
            phonemes_idx = np.asarray(p_to_i(phonemes))
            dataset[split][i]["phonemes"] = phonemes
            dataset[split][i]["phonemes_idx"] = phonemes_idx
            dpi, _ = uni2bigram(
                phonelist=phonemes_idx, phone_len=len(phonemes_idx), insert_selfloop=True,
            )
            dataset[split][i]["diphones_idx"] = np.asarray(dpi) # 1...1600
            
    return dataset


def create_llm_labels(
    dataset:    Dict[str,List[Dict[str,Any]]], 
    tokenizer:  PreTrainedTokenizer,
    prompt:     Optional[str] = "neural activity:#-> sentence:",
) -> Dict[str,List[Dict[str,Any]]]:

    prompt_tokens_a = tokenizer(prompt.split("#")[0], return_tensors="np")["input_ids"][0]
    prompt_tokens_b = tokenizer(prompt.split("#")[1], return_tensors="np")["input_ids"][0]

    for split in dataset:
        for i, row in enumerate(dataset[split]):
            if (row["sentence"] is None) and (split == "holdout"):
                row["sentence"] = "hold out"
            dataset[split][i]["input_ids"] = np.concatenate(
                (
                    prompt_tokens_a, 
                    prompt_tokens_b, 
                    tokenizer(row["sentence"] + tokenizer.eos_token, return_tensors="np")["input_ids"][0]
                ), axis=0
            )
            dataset[split][i]["attention_mask"] = np.ones_like(dataset[split][i]["input_ids"])
            dataset[split][i]["input_split"] = np.atleast_1d(prompt_tokens_a.shape[0])
            dataset[split][i]["labels"] = np.concatenate(
                (
                    np.ones_like(prompt_tokens_a)*(-100), 
                    np.ones_like(prompt_tokens_b)*(-100), 
                    tokenizer(row["sentence"] + tokenizer.eos_token, return_tensors="np")["input_ids"][0]
                ), axis=0
            )
    return dataset

    