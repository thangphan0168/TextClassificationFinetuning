"""
Fine-tune a text classifier with HuggingFace Transformers.

Usage
-----
# CSV dataset
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --text_col review \
    --num_labels 5 \
    --metric_for_best_model f1

# Freeze the backbone (only train the classification head)
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --freeze_backbone
"""

import argparse
import os
import logging

import numpy as np
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a HuggingFace model for text classification."
    )

    # Model
    p.add_argument("--model", default="distilbert/distilbert-base-uncased",
                   help="HuggingFace model name or local path (default: distilbert-base-uncased)")

    # Dataset
    p.add_argument("--train_file", default=None, required=True,
                   help="Path to a local training file")
    p.add_argument("--val_file", default=None, required=True,
                   help="Path to a local validation file")
    p.add_argument("--file_type", default="csv", required=False,
                   help="File type: csv | parquet (default: csv)")

    # Columns
    p.add_argument("--text_col", default="text",
                   help="Name of the text column in your dataset (default: text)")
    p.add_argument("--label_col", default="label",
                   help="Name of the label column in your dataset (default: label)")

    # Model settings
    p.add_argument("--num_labels", type=int, default=2,
                   help="Number of classes (default: 2)")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max token length (default: 512)")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs (default: 3)")
    p.add_argument("--train_batch_size", type=int, default=8,
                   help="Per-device training batch size (default: 8)")
    p.add_argument("--eval_batch_size", type=int, default=8,
                   help="Per-device eval batch size (default: 8)")
    p.add_argument("--backbone_lr", type=float, default=1e-5,
                   help="Learning rate for backbone (default: 1e-5); ignored when --freeze_backbone is set")
    p.add_argument("--head_lr", type=float, default=5e-5,
                   help="Learning rate for classification head (default: 5e-5)")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Warmup steps (default: 0)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze all backbone weights and train only the classification head")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducibility (default: 0)")

    # Output and logging
    p.add_argument("--output_dir", default="./results",
                   help="Directory to save the model (default: ./results)")
    p.add_argument("--logging_dir", default="./logs",
                   help="Directory for training logs (default: ./logs)")
    p.add_argument("--logging_steps", type=int, default=1,
                   help="Log every N steps (default: 1)")
    p.add_argument("--report_to", default="none",
                   help="Tracking backend: none | wandb | (default: none)")
    p.add_argument("--eval_steps", type=int, default=500,
                   help="Eval every N steps (default: 500)")
    p.add_argument("--save_steps", type=int, default=500,
                   help="Save every N steps (default: 500)")
    p.add_argument("--save_total_limit", type=int, default=3,
                   help="Total number of checkpoints to save (default: 3)")
    
    # Evaluation metric setting
    p.add_argument("--metric_for_best_model", default="accuracy", type=str,
                   help="Metric used to select the best model (options: accuracy, precision, recall, f1) (default: accuracy)")
    
    # Wandb setting
    p.add_argument("--run_name", default=None,
                   help="Experiment/run name for the logger (e.g., W&B)")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project name (optional; or set WANDB_PROJECT env var)")

    return p.parse_args()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, text_col, label_col, max_length):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item[self.text_col],
            truncation=True,
            max_length=self.max_length,
        )
        encoding["labels"] = item[self.label_col]
        return encoding


class CustomTrainer(Trainer):
    def log(self, logs, start_time=None):
        if self.optimizer is not None and isinstance(self.optimizer, torch.optim.Optimizer):
            logs["lr/head"] = self.optimizer.param_groups[0]["lr"]
            if len(self.optimizer.param_groups) > 1:
                logs["lr/backbone"] = self.optimizer.param_groups[1]["lr"]

        super().log(logs, start_time=start_time)


def freeze_backbone(model) -> int:
    """Freeze all backbone parameters and return the count of frozen params."""
    frozen = 0
    for param in model.base_model.parameters():
        param.requires_grad = False
        frozen += param.numel()
    return frozen


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.report_to == "wandb":
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Labels     : {args.num_labels}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Best metric: {args.metric_for_best_model}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*60}\n")

    print(f"Loading local CSV files: {args.train_file} / {args.val_file}")
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    dataset = load_dataset(args.file_type, data_files=data_files)
    print(f"Splits available: {list(dataset.keys())}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_split = TextDataset(dataset["train"], tokenizer, args.text_col, args.label_col, args.max_length)
    eval_split  = TextDataset(dataset["validation"], tokenizer, args.text_col, args.label_col, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.num_labels,
    )

    if args.freeze_backbone:
        frozen_count = freeze_backbone(model)
        print(f"Backbone frozen: {frozen_count:,} parameters will not be updated.\n")
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=args.head_lr,
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": args.head_lr},
            {"params": model.base_model.parameters(), "lr": args.backbone_lr},
        ])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),
        run_name=args.run_name,
        report_to=args.report_to,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")

    metrics = trainer.evaluate()
    print("\nFinal eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.report_to == "wandb":
        wandb.config.update(vars(args), allow_val_change=True)

if __name__ == "__main__":
    main()
