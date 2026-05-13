## BraIn-to-Text (BIT)

Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis by directly translating neural activity into sentences. Most existing systems use cascaded frameworks that first decode phonemes before assembling sentences with an n-gram language model. We instead introduce an end-to-end BraIn-to-Text (BIT) framework that directly translates neural activity into coherent sentences by integrating a neural encoder with audio large language models (LLMs).

- [Paper](https://arxiv.org/abs/2511.21740)
- [Setup](#environment-setup)
- [Training](#training)
- [Eval](#eval)

---

### Environment Setup

```bash
conda env create -f env.yaml
```

### Data Setup

#### Brain2Text 25

Download data from [DRYAD](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) and rename it to `brandman_2024_text`.

#### Brain2Text 24

Download `competitionData.tar.gz` from [DRYAD](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) and rename it to `willett_2023_text`.

---

### Training

*NOTE*: Update trainer YAML to use your own data and checkpoint path. For example, change the following entries in `configs/finetune/phoneme/ndt/trainer.yaml`:

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
<table>
<thead><tr>
<th><sub>Argument</sub></th><th><sub>Choices</sub></th><th><sub>Default</sub></th>
</tr></thead>
<tbody>
<tr><td><sub><code>--training_mode</code></sub></td><td><sub><code>train_from_scratch</code>, <code>finetune</code></sub></td><td><sub><code>train_from_scratch</code></sub></td></tr>
<tr><td><sub><code>--encoder</code></sub></td><td><sub><code>ndt</code></sub></td><td><sub><code>ndt</code></sub></td></tr>
<tr><td><sub><code>--task</code></sub></td><td><sub><code>none</code>, <code>phoneme</code>, <code>sentence</code></sub></td><td><sub><code>none</code></sub></td></tr>
<tr><td><sub><code>--dataset</code></sub></td><td><sub><code>none</code>, <code>willett_2023_text</code>, <code>brandman_2024_text</code></sub></td><td><sub><code>none</code></sub></td></tr>
<tr><td><sub><code>--features</code></sub></td><td><sub><code>none</code>, <code>all</code>, <code>tx1</code>, <code>spikePow</code></sub></td><td><sub><code>all</code></sub></td></tr>
<tr><td><sub><code>--ft_ckpt</code></sub></td><td><sub>Optional path to fine-tuned checkpoint</sub></td><td><sub><code>none</code></sub></td></tr>
<tr><td><sub><code>--ds_config</code></sub></td><td><sub>Optional path to DeepSpeed config</sub></td><td><sub><code>none</code></sub></td></tr>
<tr><td><sub><code>--kwargs</code></sub></td><td><sub>Additional key=value overrides</sub></td><td><sub>—</sub></td></tr>
</tbody>
</table>

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

---

### Eval

Once you have the fine-tuned model, you can generate sentence predictions in two stages:

1. Run the following command to predict phonemes:

```bash
python eval_phoneme.py --model_path YOUR_MODEL_PATH --eval_split val
```
<table>
<thead><tr>
<th><sub>Argument</sub></th><th><sub>Choices</sub></th><th><sub>Default</sub></th>
</tr></thead>
<tbody>
<tr><td><sub><code>--model_path</code></sub></td><td><sub>Path to trained model</sub></td><td><sub><code>./</code></sub></td></tr>
<tr><td><sub><code>--eval_split</code></sub></td><td><sub><code>val</code>, <code>test</code>, <code>holdout</code></sub></td><td><sub><code>test</code></sub></td></tr>
<tr><td><sub><code>--gpu_number</code></sub></td><td><sub>GPU device index</sub></td><td><sub><code>0</code></sub></td></tr>
<tr><td><sub><code>--trainer_config_path</code></sub></td><td><sub>Path to trainer config</sub></td><td><sub><code>none</code></sub></td></tr>
</tbody>
</table>

> **NOTE**: `val` specifies the validation partition, which corresponds to the test set provided by the benchmark. Use `holdout` for the holdout set of the competition. The above script outputs `{eval_split}_phoneme_logits.pt`, which can optionally be used for language model rescoring with nucleus sampling in the next step.

2. Run the following command to predict sentences using an LLM:

```bash
python eval_llm.py --model_path YOUR_MODEL_PATH --eval_split val
```
<table>
<thead><tr>
<th><sub>Argument</sub></th><th><sub>Choices</sub></th><th><sub>Default</sub></th>
</tr></thead>
<tbody>
<tr><td><sub><code>--model_path</code></sub></td><td><sub>Path to LLM model</sub></td><td><sub><code>./</code></sub></td></tr>
<tr><td><sub><code>--eval_split</code></sub></td><td><sub><code>val</code>, <code>test</code>, <code>holdout</code></sub></td><td><sub><code>test</code></sub></td></tr>
<tr><td><sub><code>--trainer_config_path</code></sub></td><td><sub>Path to trainer config</sub></td><td><sub><code>none</code></sub></td></tr>
<tr><td><sub><code>--gpu_number</code></sub></td><td><sub>GPU device index</sub></td><td><sub><code>0</code></sub></td></tr>
<tr><td><sub><code>--nbest</code></sub></td><td><sub>(Optional) number of candidate sentences for nucleus sampling</sub></td><td><sub><code>0</code></sub></td></tr>
<tr><td><sub><code>--phoneme_logits_path</code></sub></td><td><sub>(Optional) path to saved phoneme logits</sub></td><td><sub><code>none</code></sub></td></tr>
</tbody>
</table>


### Citation
Please cite our paper if you use this code in your own work:
```
@inproceedings{zhangcross,
  title={A cross-species neural foundation model for end-to-end speech decoding},
  author={Zhang, Yizi and He, Linyang and Fan, Chaofei and Liu, Tingkai and Yu, Han and Le, Trung and Li, Jingyuan and Linderman, Scott and Duncker, Lea and Willett, Francis R and others},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```