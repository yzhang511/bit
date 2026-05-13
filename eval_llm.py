import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Any

import torch
from functools import partial
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.nn import CTCLoss

from transformers import AutoTokenizer

import re

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test", "holdout"])
    parser.add_argument("--phoneme_logits_path", type=str, default="none")
    parser.add_argument("--trainer_config_path", type=str, default="none")
    parser.add_argument("--gpu_number", type=int, default=0)
    parser.add_argument("--nbest", type=int, default=0)
    return parser.parse_args()

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

def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence

def compute_wer(lm_results):

    import editdistance

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


def save_results(lm_results, model_path, eval_split):
    output_file = os.path.join(model_path, f"baseline_{eval_split}_predicted_sentences_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    df = pd.DataFrame({"id": list(range(len(lm_results["pred_sentence"]))), "text": lm_results["pred_sentence"]})
    df.to_csv(output_file, index=False)


def load_model_safetensors(model, checkpoint_dir, device="cpu", strict=True):
    from safetensors.torch import load_file as load_safetensors
    model_path = f"{checkpoint_dir}/model.safetensors"
    state_dict = load_safetensors(model_path, device=device)
    extra_keys = model.load_state_dict(state_dict, strict=strict)
    print("Missing keys:", extra_keys.missing_keys)
    print("Unexpected keys:", extra_keys.unexpected_keys)
    return model

def disable_masking(model):
    model.neural_encoder.encoder.embedder.mask_active = False
    model.neural_encoder.encoder.embedder.mask_ratio = 0.0
    model.neural_encoder.encoder.smooth_and_noise.noise = False
    model.neural_encoder.encoder.smooth_and_noise.smooth = 2.0

def maybe_load_bci(args, nbest_path, sentence_label_path):

    import json

    from safetensors.torch import load_file as load_safetensors

    PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.abspath(PROJ_DIR))

    from models.bci import BCI_AudioLLM

    from utils.config import DictConfig

    from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn

    from utils.eval_phoneme import get_model_inputs
    from utils.eval_llm import format_ctc
    from g2p_en import G2p

    NAME2MODEL = {"BCI_AudioLLM": BCI_AudioLLM}
    DATA_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, None: torch.float32}

    def load_bci_model(model_path, dataset):

        from utils.eval_llm import wer
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
        
        metric_fns = {"WER": wer}

        trainer = Trainer(
            config, dataset=dataset, metric_fns=metric_fns, 
            ds_config=ds_config, 
            extra_model_kwargs={"features": parsed["features"]},
            eval_mode=True,
        )

        # Load model
        model = load_model_safetensors(trainer.model, model_path, strict=False)
        
        disable_masking(model)
        
        return model
    

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
        

    def compute_rnn_log_probs(
            phoneme_logits: np.ndarray, sentence: str
        ) -> float:

        blank_id = 0
        
        def s_to_p(s: str):
            g2p = G2p()
            # keep only phonemes and add SIL at the end so that every word ends in SIL
            return [re.sub(r'[0-9]','',pp) if pp != " " else "SIL" for pp in g2p(s) if re.match(r'[A-Z]+', pp) or pp == " "] + ["SIL"] 
            
        def p_to_i(p: list[str]) -> list[int]:
            vocab = json.load(open("vocab.json", "r"))
            return [vocab.index(pp) for pp in p]

        phonemes = s_to_p(sentence)
        phonemes_idx = np.asarray(p_to_i(phonemes))

        phoneme_logits = torch.tensor(phoneme_logits)
        log_probs = F.log_softmax(phoneme_logits, dim=-1) 

        targets = torch.tensor(phonemes_idx, dtype=torch.long)

        input_length = torch.tensor([log_probs.size(0)], dtype=torch.long)
        target_length = torch.tensor([len(targets)], dtype=torch.long)

        # CTCLoss returns the negative log likelihood
        ctc_loss_fn = CTCLoss(blank=blank_id, reduction="none", zero_infinity=True)

        loss = - ctc_loss_fn(log_probs, targets, input_length, target_length)  # shape: (1,)

        return loss.item()


    def compute_llm_log_probs(
            scores: torch.Tensor, 
            labels: torch.Tensor, 
            trial_idx: int,
            unk_token_id: int,
        ) -> torch.Tensor:

        # token_logprob: (T, vocab_size)
        # labels: (T,)
        scores = torch.stack([score[trial_idx] for score in scores])
        token_logprob = scores.log_softmax(dim=-1)  # Ensure it's log-probabilities
        token_logprob = token_logprob.gather(1, labels.unsqueeze(-1)).squeeze(-1)

        # NOTE Skip special tokens and padding
        mask = (labels < unk_token_id)
        token_logprob = token_logprob * mask

        logprobs = token_logprob.sum()  # total log-likelihood per sequence

        return logprobs.item()
    
    from transformers import LogitsProcessor

    class RestrictVocabLogitsProcessor(LogitsProcessor):
        def __init__(self, allowed_token_ids):
            self.allowed_token_ids = set(allowed_token_ids)

        def __call__(self, input_ids, scores):
            # scores: [batch_size, vocab_size]
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[:, list(self.allowed_token_ids)] = False  # False = allowed tokens
            scores = scores.masked_fill(mask, -float('inf'))
            return scores
    
    def generate_nbest(
        model, 
        dataset, 
        dataloader, 
        tokenizer, 
        phoneme_logits,
        gen_config, 
        eval_split, 
        device, 
        data_dtype
    ):

        total_test_trials = len(dataset[eval_split])
        print(f"Total number of {eval_split} trials: {total_test_trials}\n")

        model = model.to(device=device, dtype=data_dtype)
        model.eval()

        nbest_out = []; sentence_label = []
        for trial_idx, (model_inputs, unused_inputs) in tqdm(
            enumerate(dataloader), total=total_test_trials, desc="Generate sentences"
        ):  
            
            for k, v in model_inputs.items():
                if torch.is_tensor(v):
                    if v.dtype in list(DATA_DTYPE.values()):
                        model_inputs[k] = v.to(device=device, dtype=data_dtype)
                    else:
                        model_inputs[k] = v.to(device=device)
                else:
                    model_inputs[k] = v

            # NOTE extract prompt to generate the sentence
            unk_token_id = tokenizer.vocab_size
            gen_config["pad_token_id"] = unk_token_id
            
            if "num_beams" in gen_config:
                prompt_mask = torch.logical_and(
                    model_inputs["targets"] == -100, model_inputs["input_ids"] != unk_token_id
                )
                prompt_ids = model_inputs["input_ids"][prompt_mask]
            else:
                prompt_ids = model_inputs["input_ids"]

            if len(prompt_ids.size()) == 1:
                prompt_ids = prompt_ids.unsqueeze(0)

            attention_mask = torch.ones_like(prompt_ids)
            model_inputs.update({
                "input_ids": prompt_ids, "attention_mask": attention_mask
            })

            if args.nbest > 0:
                model_inputs.pop("targets")

            if args.nbest > 0:
                # processor = RestrictVocabLogitsProcessor(allowed_token_ids=allowed_tokens)
                outputs = model.generate(
                    **model_inputs, 
                    # logits_processor=[processor],
                    **gen_config
                )
                scores = outputs.scores
                outputs = outputs.sequences
            else:
                with torch.no_grad():
                    outputs = model(**model_inputs)
                preds = outputs.preds.argmax(-1)[:,:-1]
                mask = (outputs.targets[:,1:] != -100)
                outputs = [preds[mask].squeeze(0)]

            # NOTE output acoustic score and language model score for rescoring: 
            #      acoustic score is used to weight encoder output (phonemes) probabilities
            #      lm score is used to weight the language model probabilities
            #      opt is used to re-rank the weighted combination of these two scores
            nbest = []
            for d_i, d in enumerate(outputs):
                if "num_beams" in gen_config:
                    lm_score = compute_llm_log_probs(scores, d, d_i, unk_token_id)
                else:
                    lm_score = None
                if phoneme_logits is not None:
                    ac_score = compute_rnn_log_probs(phoneme_logits[trial_idx], sentence)
                else:
                    ac_score = None
                sentence = tokenizer.decode(d, skip_special_tokens=True)
                nbest.append([sentence, ac_score, lm_score])

            print("sentence:")
            print(sentence)

            nbest_out.append(nbest)

            sentence_label.append(unused_inputs["sentence"][0])
            
        return nbest_out, sentence_label, total_test_trials
    
    device = setup_device(args.gpu_number)

    is_nbest_precomputed = os.path.isfile(nbest_path)

    if not is_nbest_precomputed:

        print("Generating nbest outputs...")
        
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

        model = load_bci_model(args.model_path, dataset)

        model_inputs = get_model_inputs(model)

        dataloader = build_dataloaders(dataset, trainer_config, args.eval_split, model_inputs)

        tokenizer = AutoTokenizer.from_pretrained(
            trainer_config.data.tokenizer_path, add_bos_token=False, add_eos_token=False
        )

        n_beams = args.nbest
        if n_beams > 0:
            gen_config = {
                "max_new_tokens": 20, 
                "do_sample": False, 
                "num_beams": n_beams, 
                "num_beam_groups": 1,      # Optional: set < n_beams if you want clustering of outputs
                "diversity_penalty": 0.0,  # > 0 if using group beam search
                "repetition_penalty": 1.0, 
                "length_penalty": 1.0, 
                "renormalize_logits": True, 
                "low_memory": True,
                "num_return_sequences": n_beams, 
                "output_scores": True, 
                "return_dict_in_generate": True,
            }
        else:
            gen_config = {
                "max_new_tokens": 20, 
                "do_sample": False,
                "low_memory": True,
            }
        
        # NOTE Load precomputed phoneme logits for language model rescoring if provided
        if args.phoneme_logits_path != "none":
            assert os.path.isfile(args.phoneme_logits_path), "Phoneme logits file does not exist"
            save_path = os.path.join(args.phoneme_logits_path, f"{args.eval_split}_phoneme_logits.npy")
            _data, _total_trials = np.load(save_path, allow_pickle=True)
            phoneme_logits = [logits[0] for logits in _data["logits"]]
            total_trials = _total_trials
        else:
            phoneme_logits = None
            total_trials = len(dataset[args.eval_split])

        nbest_out, sentence_label, total_trials = generate_nbest(
            model, 
            dataset, 
            dataloader, 
            tokenizer, 
            phoneme_logits, 
            gen_config, 
            args.eval_split, 
            device, 
            data_dtype
        )
        np.save(nbest_path, nbest_out)
        np.save(sentence_label_path, sentence_label)

        lm_results = {"true_sentence": [], "pred_sentence": []}
        for i in range(total_trials):
            lm_results["true_sentence"].append(
                sentence_label[i] if args.eval_split in ["val", "test"] else None
            )
            lm_results["pred_sentence"].append(nbest_out[i][0][0])

        if args.eval_split in ["val", "test"]:
            print("\nWER without language model rescoring:")
            compute_wer(lm_results)

        save_results(lm_results, args.model_path, args.eval_split)


def main():
    args = parse_arguments()

    nbest_path = os.path.join(args.model_path, f"{args.eval_split}_nbest.npy")
    sentence_label_path = nbest_path.replace("nbest", "sentence_label")

    maybe_load_bci(args, nbest_path, sentence_label_path)

if __name__ == "__main__":
    main()
        