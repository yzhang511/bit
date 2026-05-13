
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

| Argument           | Choices / Value                                        | Default              |
|--------------------|--------------------------------------------------------|----------------------|
| `--training_mode`  | `train_from_scratch`, `finetune`                       | `train_from_scratch` |
| `--encoder`        | `ndt`                                                  | `ndt`                |
| `--task`           | `none`, `phoneme`, `sentence`                          | `none`               |
| `--dataset`        | `none`, `willett_2023_text`, `brandman_2024_text`      | `none`               |
| `--features`       | `none`, `all`, `tx1`, `spikePow`                       | `all`                |
| `--ft_ckpt`        | Optional path to fine-tuned checkpoint                 | `none`               |
| `--ds_config`      | Optional path to DeepSpeed config                      | `none`               |
| `--kwargs`         | Additional key=value overrides                         | —                    |

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

| Argument                | Choices / Value                     | Default  |
|-------------------------|-------------------------------------|----------|
| `--model_path`          | Path to trained model               | `./`     |
| `--eval_split`           | `val`, `test`, `holdout`           | `test`   |
| `--gpu_number`          | GPU device index                    | `0`      |
| `--trainer_config_path` | Path to trainer config              | `none`   |

> **NOTE**: `val` specifies the validation partition, which corresponds to the test set provided by the benchmark. Use `holdout` for the holdout set of the competition. The above script outputs `{eval_split}_phoneme_logits.pt`, which can optionally be used for language model rescoring with nucleus sampling in the next step.

2. Run the following command to predict sentences using an LLM:

```bash
python eval_llm.py --model_path YOUR_MODEL_PATH --eval_split val
```

| Argument                | Choices / Value                                    | Default  |
|-------------------------|----------------------------------------------------|----------|
| `--model_path`          | Path to LLM model                                  | `./`     |
| `--eval_split`          | `val`, `test`, `holdout`                           | `test`   |
| `--trainer_config_path` | Path to trainer config                             | `none`   |
| `--gpu_number`          | GPU device index                                   | `0`      |
| `--nbest`               | (Optional) number of candidate sentences for nucleus sampling | `0`      |
| `--phoneme_logits_path` | (Optional) path to saved phoneme logits                       | `none`     |
