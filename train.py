import os
import json
import yaml
import argparse

import numpy as np
import torch

from transformers import AutoTokenizer

from utils.config import ParseKwargs, DictConfig, ConfigBuilder
from utils.utils import str_to_bool
from models.trainer import Trainer

from registry import dataset_registry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_mode", type=str, default="train_from_scratch",
        choices=["train_from_scratch", "finetune"],
    )
    parser.add_argument(
        "--encoder", type=str, default="ndt", choices=["ndt"]
    )
    parser.add_argument(
        "--task", type=str, default="none", choices=["none", "phoneme", "sentence"]
    )
    parser.add_argument(
        "--dataset", type=str, default="none", 
        choices = ["none", "willett_2023_text", "brandman_2024_text"]
    )
    parser.add_argument(
        "--features", type=str, default="all", choices=["none", "all", "tx1", "spikePow"]
    )
    parser.add_argument("--ft_ckpt", type=str, default="none")
    parser.add_argument("--ds_config", type=str, default="none")
    parser.add_argument("--kwargs", nargs="*", action=ParseKwargs)
    return parser.parse_args()

def main(args):

    builder = ConfigBuilder(args)
    config, ds_config = builder.build()

    def load_brain2text_dataset(data_config):
        from data.willett_2023_text.prepare_data import create_phonemes_ctc_labels, create_llm_labels
        dataset_name = config.method.dataset_kwargs.dataset_name
        data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/{dataset_name}"
        module_path = dataset_registry[dataset_name]
        module = __import__(module_path, fromlist=["load_data"])
        dataset = module.load_data(**data_config.dataset)
        if "vocab_file" in data_config and data_config.vocab_file is not None:
            if "holdout" in dataset:
                dataset.pop("holdout")  # remove test split since it has no labels
            dataset = create_phonemes_ctc_labels(dataset, data_config.vocab_file)
        if "tokenizer_path" in data_config and data_config.tokenizer_path is not None:
            if "holdout" in dataset:
                dataset.pop("holdout")  # remove test split since it has no labels
            tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_path, add_bos_token=False, add_eos_token=False)
            dataset = create_llm_labels(dataset, tokenizer, data_config.prompt)
        return dataset

    if config.data.dataset_class in ["ssl", "sl"]:
        dataset = load_brain2text_dataset(config.data)
    else:
        raise ValueError(f"Unsupported dataset class: {config.data.dataset_class}")
    
    metric_fns = {}

    if config.method.model_kwargs.method_name in ["ssl"]:
        from utils.eval_spike import eval_neurons_metric
        metric_name = "r2" if config.method.model_kwargs.loss == "mse" else "bps"
        metric_fns.update({metric_name: eval_neurons_metric})
    elif config.method.model_kwargs.method_name == "ctc":
        from utils.eval_llm import per
        metric_fns.update({"PER": per})
    elif config.method.model_kwargs.method_name == "endtoend":
        from utils.eval_llm import wer
        metric_fns.update({"WER": wer})
    else:
        raise ValueError(f"Unsupported method name: {config.method.model_kwargs.method_name}")

    if metric_fns == {}:
        metric_fns = None

    trainer = Trainer(
        config, 
        dataset=dataset, 
        metric_fns=metric_fns, 
        ds_config=ds_config, 
        extra_model_kwargs={"features": args.features},
    )
    trainer.train()
    

if __name__ == "__main__":
    
    args = parse_arguments()

    main(args)
    