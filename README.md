## A cross-species neural foundation model for end-to-end speech decoding

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

#### Training Args

<sub><table><thead><tr>
<th>Argument</th><th>Choices</th><th>Default</th>
</tr></thead><tbody>
<tr><td><code>--training_mode</code></td><td><code>train_from_scratch</code>, <code>finetune</code></td><td><code>train_from_scratch</code></td></tr>
<tr><td><code>--encoder</code></td><td><code>ndt</code></td><td><code>ndt</code></td></tr>
<tr><td><code>--task</code></td><td><code>none</code>, <code>phoneme</code>, <code>sentence</code></td><td><code>none</code></td></tr>
<tr><td><code>--dataset</code></td><td><code>none</code>, <code>willett_2023_text</code>, <code>brandman_2024_text</code></td><td><code>none</code></td></tr>
<tr><td><code>--features</code></td><td><code>none</code>, <code>all</code>, <code>tx1</code>, <code>spikePow</code></td><td><code>all</code></td></tr>
<tr><td><code>--ft_ckpt</code></td><td>Optional path to fine-tuned checkpoint</td><td><code>none</code></td></tr>
<tr><td><code>--ds_config</code></td><td>Optional path to DeepSpeed config</td><td><code>none</code></td></tr>
<tr><td><code>--kwargs</code></td><td>Additional key=value overrides</td><td>—</td></tr>
</tbody></table></sub>

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

<sub><table><thead><tr>
<th>Argument</th><th>Choices</th><th>Default</th>
</tr></thead><tbody>
<tr><td><code>--model_path</code></td><td>Path to trained model</td><td><code>./</code></td></tr>
<tr><td><code>--eval_split</code></td><td><code>val</code>, <code>test</code>, <code>holdout</code></td><td><code>test</code></td></tr>
<tr><td><code>--gpu_number</code></td><td>GPU device index</td><td><code>0</code></td></tr>
<tr><td><code>--trainer_config_path</code></td><td>Path to trainer config</td><td><code>none</code></td></tr>
</tbody></table></sub>

> **NOTE**: `val` specifies the validation partition, which corresponds to the test set provided by the benchmark. Use `holdout` for the holdout set of the competition. The above script outputs `{eval_split}_phoneme_logits.pt`, which can optionally be used for language model rescoring with nucleus sampling in the next step.

2. Run the following command to predict sentences using an LLM:

```bash
python eval_llm.py --model_path YOUR_MODEL_PATH --eval_split val
```

<sub><table><thead><tr>
<th>Argument</th><th>Choices</th><th>Default</th>
</tr></thead><tbody>
<tr><td><code>--model_path</code></td><td>Path to LLM model</td><td><code>./</code></td></tr>
<tr><td><code>--eval_split</code></td><td><code>val</code>, <code>test</code>, <code>holdout</code></td><td><code>test</code></td></tr>
<tr><td><code>--trainer_config_path</code></td><td>Path to trainer config</td><td><code>none</code></td></tr>
<tr><td><code>--gpu_number</code></td><td>GPU device index</td><td><code>0</code></td></tr>
<tr><td><code>--nbest</code></td><td>(Optional) number of candidate sentences for nucleus sampling</td><td><code>0</code></td></tr>
<tr><td><code>--phoneme_logits_path</code></td><td>(Optional) path to saved phoneme logits</td><td><code>none</code></td></tr>
</tbody></table></sub>


### Citation
Please cite our paper if you use this code in your own work:
```
@inproceedings{zhangcross,
  title={A cross-species neural foundation model for end-to-end speech decoding},
  author={Zhang, Yizi and He, Linyang and Fan, Chaofei and Liu, Tingkai and Yu, Han and Le, Trung and Li, Jingyuan and Linderman, Scott and Duncker, Lea and Willett, Francis R and others},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```