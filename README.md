# Text Classifier Fine-Tuning

Fine-tune any HuggingFace transformer model for text classification using a simple CLI. Supports CSV and Parquet datasets, differential learning rates, backbone freezing, and Weights & Biases logging.

---

## Motivation
This repository is created to explore the proper way to fine tune a text classification in various scenarios: binary classification, binary classification with imbalanced data and multi-class classification.

Most guide to fine tuning text classification nowadays usually just apply a randomly initialized linear layer on top of a pre-trained language model then fine tune all layers with the same learning rate. (E.g. https://huggingface.co/docs/transformers/en/tasks/sequence_classification). Inspired by this [blog](https://lalatenduswain.medium.com/fine-tuning-pre-trained-models-the-right-way-a-step-by-step-guide-to-learning-rate-strategy-b3d9c0307222), I want to investigate the effect of better hypeparameter tuning, especially the **freeze and unfreeze backbone method** and **differential learning rate** on the performance on various text classification tasks. Experiments will be keep tracked in [experiments.md](experiments.md).

Datasets planned to be used as experiments:
- [IMDB review dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Email spam dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
- [20K News Groups](https://huggingface.co/datasets/SetFit/20_newsgroups)

---
> The following section was generated with AI assistance.
## Requirements

```bash
pip install torch transformers datasets scikit-learn wandb
```

---

## Quick Start

### Binary classification (default)

```bash
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv
```

### Multi-class classification

```bash
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --text_col review \
    --label_col sentiment \
    --num_labels 5
```

### Freeze the backbone — only train the classification head

```bash
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --freeze_backbone \
    --head_lr 5e-5
```

### Log to Weights & Biases

```bash
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --report_to wandb \
    --wandb_project my-project \
    --run_name experiment-01
```

### Use a different backbone (e.g. RoBERTa)

```bash
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --model roberta-base \
    --num_labels 3 \
    --epochs 5 \
    --train_batch_size 16
```

---

## Dataset Format

Your CSV (or Parquet) file must have at least two columns — one for the text and one for the integer label. Column names are configurable via `--text_col` and `--label_col`.

**Example `train.csv`:**

```
text,label
"I loved this product, highly recommend!",1
"Terrible experience, would not buy again.",0
"Average quality, nothing special.",0
```

Labels must be **zero-indexed integers** (0, 1, 2, …).

---

## Arguments

### Model

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `distilbert/distilbert-base-uncased` | HuggingFace model name or path to a local checkpoint |

### Dataset

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train_file` | `str` | *(required)* | Path to the training file |
| `--val_file` | `str` | *(required)* | Path to the validation file |
| `--file_type` | `str` | `csv` | File format: `csv` or `parquet` |

### Columns

| Argument | Type | Default | Description |
|---|---|---|---|
| `--text_col` | `str` | `text` | Name of the column containing input text |
| `--label_col` | `str` | `label` | Name of the column containing integer class labels |

### Model Settings

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_labels` | `int` | `2` | Number of output classes |
| `--max_length` | `int` | `512` | Maximum token length; longer sequences are truncated |

### Training Hyperparameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `3` | Number of full passes over the training data |
| `--train_batch_size` | `int` | `8` | Per-device batch size during training |
| `--eval_batch_size` | `int` | `8` | Per-device batch size during evaluation |
| `--backbone_lr` | `float` | `1e-5` | Learning rate for the transformer backbone (ignored when `--freeze_backbone` is set) |
| `--head_lr` | `float` | `5e-5` | Learning rate for the classification head |
| `--warmup_steps` | `int` | `0` | Number of linear warmup steps before the cosine schedule kicks in |
| `--freeze_backbone` | flag | `False` | Freeze all backbone weights; only the classification head is trained |

### Output & Checkpointing

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output_dir` | `str` | `./results` | Directory where the best model and tokenizer are saved |
| `--logging_dir` | `str` | `./logs` | Directory for TensorBoard / training logs |
| `--logging_steps` | `int` | `1` | Log training metrics every N optimizer steps |
| `--eval_steps` | `int` | `500` | Run evaluation every N steps |
| `--save_steps` | `int` | `500` | Save a checkpoint every N steps |
| `--save_total_limit` | `int` | `3` | Maximum number of checkpoints to keep on disk |

### Experiment Tracking

| Argument | Type | Default | Description |
|---|---|---|---|
| `--report_to` | `str` | `none` | Tracking backend: `none` or `wandb` |
| `--run_name` | `str` | `None` | Display name for the run in the tracking UI |
| `--wandb_project` | `str` | `None` | W&B project name (can also be set via the `WANDB_PROJECT` environment variable) |

---

## Output

After training completes the following are written to `--output_dir`:

- `pytorch_model.bin` — best checkpoint (highest validation accuracy)
- `config.json` — model configuration
- `tokenizer_config.json` + vocab files — tokenizer

Final evaluation metrics are printed to stdout.

---

## Notes

- **Differential learning rates** — when the backbone is not frozen, the head trains at `--head_lr` and the backbone at `--backbone_lr`. Both learning rates are logged separately (`lr/head`, `lr/backbone`) at every step.
- **Mixed precision** — `fp16` is enabled automatically when a CUDA GPU is detected.
- **Best model** — `load_best_model_at_end=True` ensures the checkpoint with the highest accuracy is what gets saved, not the last one.
- **LR schedule** — cosine decay is used by default, with an optional linear warmup controlled by `--warmup_steps`.
