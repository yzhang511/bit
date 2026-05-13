import os
from copy import deepcopy
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import scipy
import numpy as np
import math
import random
from pathlib import Path

from torch.utils.data import Dataset, Sampler

PAD_VALUE = -100

DATASET_TO_IDX = {
    "brandman_2024_text": 0,
    "willett_2023_text": 1,
}

DATASET_CHANNELS = {
    "tx1": {
        "brandman_2024_text": 256,
        "willett_2023_text": 128,
    },
    "spikePow": {
        "brandman_2024_text": 256,
        "willett_2023_text": 128,
    },
    "all": {
        "brandman_2024_text": 512,
        "willett_2023_text": 256,
    },
}

class SpikingDataset(Dataset):
    def __init__(
        self, 
        dataset:      Dict[str, List[Any]], 
        split:        str,
        length:       Optional[int] = None,
        spikes_name:  Optional[str] = "spikes",
        dataset_name: Optional[str] = "brandman_2024_text",
        **kwargs,
    ):  
        self.dataset = dataset[split]
        self.spikes_name = spikes_name
        self.dataset_name = DATASET_TO_IDX[dataset_name]
        self.dataset = self.dataset[:length] if length is not None else self.dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs = deepcopy(self.dataset[idx])

        spikes = inputs.pop(f"{self.spikes_name}")    

        inputs.update({
            "spikes": spikes,                                          
            "spikes_mask": np.ones(spikes.shape[0], dtype=np.int64),  
            "spikes_timestamp": np.arange(0, spikes.shape[0]),         
            "spikes_spacestamp": np.arange(0, spikes.shape[1]),       
            "spikes_lengths": np.asarray(spikes.shape[0]),      
            "dataset_name": np.array(self.dataset_name), 
            "day_idx": inputs["day_idx"],
            "block_idx": inputs["block_idx"],
        })
        return inputs
    

class SpikingDatasetForDecoding(SpikingDataset):
    def __init__(
        self, 
        dataset:      List[Dict[str, Union[np.ndarray, Any]]], 
        split:        str,
        length:       Optional[int] = None,
        spikes_name:  Optional[str] = "spikes",
        targets_name: Optional[str] = "targets",
        dataset_name: Optional[str] = "willett_2023_text",
        **kwargs,
    ):  
        super().__init__(dataset, split, length)
        
        self.targets_name = targets_name
        self.dataset_name = DATASET_TO_IDX[dataset_name]

    def __getitem__(self, idx):
        inputs = deepcopy(self.dataset[idx])
        
        spikes = inputs.pop(f"{self.spikes_name}")                          

        if self.targets_name == "diphones_idx":
            targets = inputs.pop("phonemes_idx") 
            targets_sub = inputs.pop("diphones_idx")
            inputs.update({
                "targets_sub":  targets_sub,
                "targets_sub_lengths": np.asarray(targets_sub.shape[0]),
            })
        else:
            targets = inputs.pop(f"{self.targets_name}") 

        inputs.update({
            "spikes": spikes,                                         
            "spikes_mask": np.ones(spikes.shape[0], dtype=np.int64),  
            "spikes_timestamp": np.arange(0,spikes.shape[0]),       
            "spikes_spacestamp": np.arange(0,spikes.shape[1]),    
            "spikes_lengths": np.asarray(spikes.shape[0]),         
            "targets":  targets,                                 
            "targets_mask": np.ones_like(targets),                
            "targets_lengths": np.asarray(targets.shape[0]),  
            "dataset_name": np.array(self.dataset_name),
            "day_idx": inputs["day_idx"],
            "block_idx": inputs["block_idx"],
        })
        return inputs


class GroupBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        # Build mapping: dataset_name -> list of indices
        self.groups = {}
        for idx in range(len(dataset)):
            name = dataset[idx]["dataset_name"]
            if hasattr(name, "item"):  # handle scalar tensors
                name = name.item()
            self.groups.setdefault(name, []).append(idx)

    def __iter__(self):
        batches = []

        for indices in self.groups.values():
            indices = np.array(indices)
            if self.shuffle:
                self.rng.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        yield from batches

    def __len__(self):
        total_batches = 0
        for indices in self.groups.values():
            total_batches += len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                total_batches += 1
        return total_batches


def padded_array(
    arrays: List[np.ndarray],
    dim: Optional[Union[int, List[int]]] = 0,
    side: Optional[Union[str, List[str]]] = "right",
    value: Optional[Union[int, List[int]]] = 0,
    truncate: Optional[Union[int, List[int]]] = None,
    min_length: Optional[Union[int, List[int]]] = None,
    max_length: Optional[Union[int, List[int]]] = None,
) -> np.ndarray:
    """Pad multiple dimensions of numpy arrays.
    
    Args:
        arrays: List of arrays to pad
        dim: Dimension(s) to pad
        side: Side(s) to pad ('left' or 'right')
        value: Padding value(s)
        truncate: Maximum length(s) after padding
        min_length: Minimum length(s) after padding
    """
    # Convert single values to lists
    dims = [dim] if isinstance(dim, int) else dim
    sides = [side] * len(dims) if isinstance(side, str) else side
    values = [value] * len(dims) if isinstance(value, (int, float)) else value
    truncates = [truncate] * len(dims) if truncate is None or isinstance(truncate, int) else truncate
    min_lengths = [min_length] * len(dims) if min_length is None or isinstance(min_length, int) else min_length
    max_lengths = [max_length] * len(dims) if max_length is None or isinstance(max_length, int) else max_length

    # Start with the original arrays
    padded_arrays = arrays

    # Apply padding for each dimension
    for dim_idx, (d, s, v, t, m, l) in enumerate(zip(dims, sides, values, truncates, min_lengths, max_lengths)):
        # Get maximum size for this dimension
        if d == 0:
            max_size = max(arr.shape[d] for arr in padded_arrays)
        elif d == 1:
            max_size = l
        if t is None:
            t = max_size
        if m is None:
            m = 0
        assert m <= t, f"Can't truncate below minimum length for dimension {d}"
        pad_size = min(t, max(max_size, m))

        # Create padding widths
        pad_width = np.zeros((padded_arrays[0].ndim, 2), dtype=np.int64)
        if s == "left":
            pad_width[d, 0] = 1
        elif s == "right":
            pad_width[d, 1] = 1
        else:
            raise ValueError(f"'side' can only be 'right' or 'left', got {s}")

        # Create slice for truncation
        slc = [slice(None)] * padded_arrays[0].ndim
        slc[d] = slice(0, t)

        # Apply padding for this dimension
        padded_arrays = [
            np.pad(
                arr, 
                pad_width * max(0, pad_size - arr.shape[d]), 
                mode="constant", 
                constant_values=v
            )[tuple(slc)] 
            for arr in padded_arrays
        ]

    return np.stack(padded_arrays, axis=0)


def pad_collate_fn(
    batch: List[Dict[str, Union[np.ndarray, Any]]], 
    model_inputs: List[str],
    pad_dict: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
) -> Tuple[Dict[str, Union[torch.Tensor, List[Union[torch.Tensor, Any]]]]]:
    """Collate function that handles multi-dimensional padding.
    
    Args:
        batch: List of examples
        model_inputs: Names of keys used by model
        pad_dict: Padding configuration for each key
    """

    def convert_dtype(tensor):
        """Convert only float64/double to float32, keep other dtypes."""
        if tensor.dtype in [torch.float64, torch.double]:
            return tensor.float()
        return tensor

    # Case when batching is done in dataset
    if isinstance(batch[0], list):
        batch = [row for sub_batch in batch for row in sub_batch]

    keys = batch[0].keys()
    array_keys = [k for k in keys if isinstance(batch[0][k], np.ndarray)]
    string_array_keys = [k for k in keys if isinstance(batch[0][k], np.ndarray) and batch[0][k].dtype.type == np.str_]
    pad_keys = pad_dict.keys()
    
    assert set(pad_keys).issubset(array_keys), f"Can't pad non-array keys: {set(pad_keys)-set(array_keys)}"
    
    padded_batch = {}
    unused_inputs = {}
    
    for key in keys:
        if key in array_keys:
            if key in pad_keys:
                # Handle padding as before
                pad_config = pad_dict[key]
                if isinstance(pad_config, dict):
                    pad_config = [pad_config]
                value = torch.from_numpy(
                    padded_array(
                        [row[key] for row in batch],
                        dim=[p["dim"] for p in pad_config],
                        side=[p["side"] for p in pad_config],
                        value=[p["value"] for p in pad_config],
                        truncate=[p.get("truncate") for p in pad_config],
                        min_length=[p.get("min_length") for p in pad_config],
                        max_length=[p.get("max_length") for p in pad_config],
                    )
                ).clone()
                value = convert_dtype(value)
            elif key in string_array_keys:
                # Keep string arrays as numpy arrays
                value = np.stack([row[key] for row in batch], axis=0)
            elif len(set(row[key].shape for row in batch)) == 1:
                # Convert numeric arrays to tensors
                value = convert_dtype(torch.from_numpy(np.stack([row[key] for row in batch], axis=0)))
            else:
                value = [convert_dtype(torch.from_numpy(row[key])) for row in batch]
        else:
            value = [row[key] for row in batch if key in row]

        if key in model_inputs:
            padded_batch[key] = value
        else:
            unused_inputs[key] = value

    return padded_batch, unused_inputs


def save_load_arrays(
    cache_paths:  Dict[str, Path], 
    mode:         str, 
    subject:      Optional[str] = "",  
    date:         Optional[str] = "", 
    experiment:   Optional[str] = "",
    data:         Optional[np.ndarray] = None, 
):

    if mode == "save":
        for path in cache_paths.values():
            os.makedirs(path, exist_ok=True)
        
        arrays = {"spike_tx": data}
        for name, array in arrays.items():
            np.save(cache_paths[name]/f"{subject}_{date}_{experiment}.npy", array)
    else:
        # load mode
        # Sort file names to avoid introducing randomness from os.listdir
        # Use os.scandir() which is more efficient than os.listdir()
        print("Loading data...")
        with os.scandir(cache_paths["spike_tx"]) as it:
            file_names = sorted(
                f.name for f in it 
                if f.name.endswith('.npy') and f.is_file()
            )            
        return [np.load(cache_paths["spike_tx"]/f, allow_pickle=True, mmap_mode="r") for f in file_names]
    
    