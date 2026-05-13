## BraIn-to-Text (BIT)

Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis by directly translating neural activity into sentences. Most existing systems use cascaded frameworks that first decode phonemes before assembling sentences with an n-gram language model. We instead introduce an end-to-end BraIn-to-Text (BIT) framework that directly translates neural activity into coherent sentences by integrating a neural encoder with audio large language models (LLMs).

- [Paper](https://arxiv.org/abs/2511.21740)
- [Setup](#environment-setup)
- [Training](#training)
- [Eval](#eval)

## Setup

```bash
conda env create -f env.yaml
```

#### Brain2Text 25

Download data from [DRYAD](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) and rename it to `brandman_2024_text`.

#### Brain2Text 24

Download `competitionData.tar.gz` from [DRYAD](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) and rename it to `willett_2023_text`.


## Training

Update trainer YAML to use your own data and checkpoint path. For example, change the following entries in `configs/finetune/phoneme/ndt/trainer.yaml`:

```yaml
dirs:
  data_dir: YOUR_DATA_DIR
  checkpoint_dir: YOUR_CHECKPOINT_DIR
  log_dir: YOUR_LOG_DIR
```

Run the following command to train a model:

```bash
python train.py --training_mode MODE \
                --dataset DATASET \
                --features FEATURES \
                --encoder ENCODER \
                --task TASK \
                [--ft_ckpt CKPT] \
                [--ds_config DS_CONFIG] \
                [--kwargs KEY=VALUE ...]
```
*NOTE*:
- `--training_mode`: `train_from_scratch`, `finetune`
- `--encoder`: `ndt`
- `--task`: `none`, `phoneme`, `sentence`
- `--dataset`: `none`, `willett_2023_text`, `brandman_2024_text`
- `--features`: `none`, `all`, `tx1`, `spikePow`
- `--ft_ckpt`: path to fine-tuned checkpoint (optional)
- `--ds_config`: path to DeepSpeed config (optional)
- `--kwargs`: additional key=value overrides (optional)

#### Example

1. Train from scratch for phoneme decoding:

```bash
python train.py --training_mode train_from_scratch \
                --dataset brandman_2024_text \
                --features all \
                --encoder ndt \
                --task phoneme
```

2. Fine-tune the above model for sentence decoding:

```bash
python train.py --training_mode finetune \
                --dataset brandman_2024_text \
                --features all \
                --encoder ndt \
                --task sentence \
                --ft_ckpt YOUR_MODEL_PATH
```

## Eval

Once you have the fine-tuned model, you can generate sentence predictions in two stages:

1. Run the following command to predict phonemes:

```bash
python eval_phoneme.py --model_path YOUR_MODEL_PATH --eval_split val
```
*NOTE*: 
- `--model_path`: path to trained model
- `--eval_split`: `val`, `test`, `holdout`
- `val` specifies the validation partition, which corresponds to the test set provided by the benchmark. Use `holdout` for the holdout set of the competition. 
- Outputs `{eval_split}_phoneme_logits.pt` that can be used for language model rescoring.

2. Run the following command to predict sentences using an LLM:

```bash
python eval_llm.py --model_path YOUR_MODEL_PATH --eval_split val
```
*NOTE*:
- `--model_path`: path to LLM model
- `--eval_split`: `val`, `test`, `holdout`
- `--nbest`: number of candidate sentences for nucleus sampling (optional) 
- `--phoneme_logits_path`: path to saved phoneme logits (optional) 


## Citation
Please cite our paper if you use this code in your own work:
```
@inproceedings{zhangcross,
  title={A cross-species neural foundation model for end-to-end speech decoding},
  author={Zhang, Yizi and He, Linyang and Fan, Chaofei and Liu, Tingkai and Yu, Han and Le, Trung and Li, Jingyuan and Linderman, Scott and Duncker, Lea and Willett, Francis R and others},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```