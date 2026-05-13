import os
import re
import json
import string

from pathlib import Path
from glob import glob

from tqdm import tqdm
from typing import List, Optional, Any, Dict, Tuple
from collections import defaultdict

import h5py

import scipy
import numpy as np

# Time bin size: 20 ms

def load_data(
    data_dir:        str, 
    day_idxs:        Optional[List[int]] = None,
    zscore_block:    Optional[bool] = False, 
    zscore_day:      Optional[bool] = False,
    features:        Optional[List[str]] = ["tx1"],  # ["tx1", "spikePow"]
    area_start:      Optional[int] = 0, 
    area_end:        Optional[int] = 256,
    **kwargs,
) -> Dict[str, List[Dict[str, Any]]]:
    
    assert not zscore_block and not zscore_day, "Neural features are already z-scored."
    
    data_dir = Path(data_dir) / "hdf5_data_final"
    sessions = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    sessions = sorted(
        sessions, key=lambda file: tuple(file.split("/")[-1].split(".")[1:4])
    )
    feat_end = 256

    def load_single_file(file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        data = load_h5py_file(file)
        y_i = data["sentence_label"]
        b_i = data["block_num"]
        d_i = tuple(file.split("/")[-2].split(".")[1:4])

        tx1 = np.array(
            [
                data["neural_features"][i][:, area_start:area_end] for i in range(len(y_i))
            ], dtype = np.ndarray
        )
        spikePow = np.array(
            [
                data["neural_features"][i][:, feat_end+area_start:feat_end+area_end] for i in range(len(y_i))
            ], dtype = np.ndarray
        )

        if "tx1" in features and "spikePow" not in features:
            x_i = tx1
        elif "tx1" not in features and "spikePow" in features:
            x_i = spikePow
        elif "tx1" in features and "spikePow" in features:
            x_i = np.array(
                [
                    np.concatenate((tx1[i], spikePow[i]), axis=-1) for i in range(len(y_i))
                ], dtype = np.ndarray
            )
        else:
            raise ValueError("Invalid feature combination. Must include either 'tx1' or 'spikePow'.")
            
        return (
            x_i, y_i, b_i, [d_i] * len(y_i)
        )

    def get_split_dict(split: str) -> List[Dict]:
        split_files = []
        for session in sessions:
            session_path = data_dir / session
            split_files.extend(
                [session_path / f for f in os.listdir(session_path) if f == f"data_{split}.hdf5"]
            )
    
        records = []
        for file in tqdm(split_files, desc=f"Loading {split}"):
            try:
                x_i, y_i, b_i, d_i = load_single_file(str(file))

                for xi, yi, bi, di in zip(x_i, y_i, b_i, d_i):
                    records.append({
                        "spikes": xi.astype(np.float32),
                        "sentence": yi.translate(str.maketrans("", "", string.punctuation.replace("'", ""))).lower().strip() if yi is not None else None,
                        "block": bi - 1,
                        "day": di,
                    })
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue
        return records

    splits = ["train", "val", "test"]
    dataset_dict = {
        split: get_split_dict(split) for split in splits
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

    dataset_dict["holdout"] = dataset_dict.pop("test")

    dataset_dict = split_val(dataset_dict)

    return dataset_dict


def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': []
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data

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
