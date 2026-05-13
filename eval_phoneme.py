import os
import sys
import numpy as np

from tqdm import tqdm
import argparse

import json
from functools import partial
from torch.utils.data import DataLoader

from models.ndt import NDT

from utils.config import DictConfig

from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn

from utils.eval_phoneme import *
from utils.eval_llm import format_ctc

NAME2MODEL = {"NDT": NDT}
DATA_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, None: torch.float32}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test", "holdout"])
    parser.add_argument("--gpu_number", type=int, default=0)
    parser.add_argument("--trainer_config_path", type=str, default="none")
    return parser.parse_args()

def load_model_safetensors(model, checkpoint_dir, device="cpu", strict=True):
    from safetensors.torch import load_file as load_safetensors
    model_path = f"{checkpoint_dir}/model.safetensors"
    state_dict = load_safetensors(model_path, device=device)
    extra_keys = model.load_state_dict(state_dict, strict=strict)
    return model

def disable_masking(encoder):
    encoder.embedder.mask_active = False
    encoder.embedder.mask_ratio = 0.0
    encoder.smooth_and_noise.noise = False
    encoder.smooth_and_noise.smooth = 2.0

def load_ndt_model(model_path, dataset, dataset_name=None, seed=42):

    from utils.eval_llm import per
    from models.trainer import Trainer
    from utils.config import ConfigBuilder

    savestring = model_path.split("/")[-2]
    parsed = ConfigBuilder.parse_savestring(savestring)
    parsed["ft_ckpt"] = "none"
    parsed["ds_config"] = "none"

    config = parsed

    builder = ConfigBuilder(config)
    config, ds_config = builder.build()
    config["log_to_wandb"] = False

    metric_fns = {"PER": per}

    trainer = Trainer(
        config, 
        dataset=dataset, 
        metric_fns=metric_fns, 
        extra_model_kwargs={"features": parsed["features"]},
        eval_mode=True,
    )
    
    # Load model
    load_model_safetensors(trainer.model, model_path, strict=False)

    disable_masking(trainer.model.encoder)

    return trainer.model


def load_dataset(config):
    from registry import dataset_registry
    from transformers import AutoTokenizer
    from data.willett_2023_text.prepare_data import create_phonemes_ctc_labels, create_llm_labels
    data_config = config.data
    dataset_name = config.method.dataset_kwargs.dataset_name
    data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/{dataset_name}"
    module_path = dataset_registry[dataset_name]
    module = __import__(module_path, fromlist=["load_data"])
    dataset = module.load_data(**data_config.dataset)
    if "vocab_file" in data_config and data_config.vocab_file is not None:
        dataset = create_phonemes_ctc_labels(dataset, data_config.vocab_file)
    if "tokenizer_path" in data_config and data_config.tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_path, add_bos_token=False, add_eos_token=False)
        dataset = create_llm_labels(dataset, tokenizer, data_config.prompt)
    return dataset

def build_dataloaders(dataset, config, eval_split, model_inputs):
    dataset_class = SpikingDatasetForDecoding

    eval_name = eval_split
    eval_len = config.data.test_len if eval_split == "test" else config.data.val_len
    dataset = dataset_class(
        dataset, eval_name, length=eval_len, **config.method.dataset_kwargs
    )
    dataloader = DataLoader(
        dataset, 
        shuffle=False, 
        collate_fn=partial(
            pad_collate_fn, model_inputs=model_inputs, **config.method.dataloader_kwargs
        ), 
        batch_size=1, 
        pin_memory=True, 
        drop_last=False,
    )
    return dataloader

def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def evaluate_model(model, dataset, dataloader, config, eval_split, device, data_dtype):
    test_data = {"logits": [], "pred_seq": [], "true_seq": [], "sentence_label": []}
    total_test_trials = len(dataset[eval_split])
    print(f"Total number of {eval_split} trials: {total_test_trials}\n")

    model = model.to(device=device, dtype=data_dtype)

    model.eval()
    with torch.no_grad():
        for trial_idx, (model_inputs, unused_inputs) in tqdm(
            enumerate(dataloader), total=total_test_trials, desc="Predict phonemes"
        ):  
            for k, v in model_inputs.items():
                if torch.is_tensor(v):
                    if v.dtype in list(DATA_DTYPE.values()):
                        model_inputs[k] = v.to(device=device, dtype=data_dtype)
                    else:
                        model_inputs[k] = v.to(device=device)
                else:
                    model_inputs[k] = v

            outputs = model(**model_inputs)
            logits = to_numpy(outputs.preds)

            test_data["logits"].append(logits)

            blank_id = config.method.model_kwargs.blank_id
            vocab = json.load(open(config.data.vocab_file,"r"))

            preds = logits.argmax(-1)
            pred_seq = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
            true_seq = [" ".join(p) for p in unused_inputs["phonemes"]]
            sentence_label = unused_inputs["sentence"] 
            test_data["pred_seq"].append(" ".join(pred_seq))
            test_data["true_seq"].append(" ".join(true_seq))
            test_data["sentence_label"].append(sentence_label[0])

    for i in range(total_test_trials):
        print(
            "\n-----\n ", 
            test_data["pred_seq"][i].replace(" ","").replace("SIL"," SIL "), 
            "\n-----\n ", 
            test_data["true_seq"][i].replace(" ","").replace("SIL"," SIL "), 
            "\n-----\n ", 
            test_data["sentence_label"][i], 
            "\n-----\n\n "
        )
        
    if eval_split != "holdout":
        from utils.eval_llm import word_error_count
        errors, n_phonemes = word_error_count(test_data["pred_seq"], test_data["true_seq"])
        print(f"{errors/n_phonemes:.4f} PER on {eval_split} set.")

    return test_data, total_test_trials


def main():
    args = parse_arguments()
    device = setup_device(args.gpu_number)

    save_path = os.path.join(args.model_path, f"{args.eval_split}_phoneme_logits.npy")
    
    if args.trainer_config_path == "none":
        trainer_config = DictConfig(
            torch.load(os.path.join(args.model_path, "trainer_config.pth"))
        )
    else:
        from utils.config import update_config
        DEFAULT_TRAINER_CONFIG = "configs/trainer.yaml"
        trainer_config = update_config(DEFAULT_TRAINER_CONFIG, args.trainer_config_path)

    mixed_precision = None
    data_dtype = DATA_DTYPE[mixed_precision]

    dataset = load_dataset(trainer_config)

    if args.trainer_config_path == "none":
        trainer_config_path = os.path.join(args.model_path, "trainer_config.pth")
    else:
        trainer_config_path = args.trainer_config_path

    if "willett_2023_text" in trainer_config_path:
        dataset_name = "willett_2023_text"
    elif "brandman_2024_text" in trainer_config_path:
        dataset_name = "brandman_2024_text"
    else:
        raise ValueError(f"Unsupported dataset: {trainer_config_path}")

    if "ndt" in args.model_path:
        model = load_ndt_model(args.model_path, dataset, dataset_name)
    else:
        raise ValueError(f"Unsupported model: {args.model_path}")

    model_inputs = get_model_inputs(model)

    dataloader = build_dataloaders(dataset, trainer_config, args.eval_split, model_inputs)
    
    test_data, total_trials = evaluate_model(
        model, dataset, dataloader, trainer_config, args.eval_split, device, data_dtype
    )
    np.save(save_path, [test_data, total_trials])
    

if __name__ == "__main__":
    main()
        