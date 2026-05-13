"""
Fine-tune a text classifier with HuggingFace Transformers.

Usage
-----
# CSV dataset
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --text_col review \
    --num_labels 5
"""

import argparse
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    get_cosine_schedule_with_warmup
)
from transformers.integrations.integration_utils import WandbCallback


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a HuggingFace model for text classification."
    )

    # Model
    p.add_argument("--model", default="distilbert/distilbert-base-uncased",
                   help="HuggingFace model name or local path (default: distilbert-base-uncased)")

    # Dataset
    p.add_argument("--train_file", default=None, required=True,
                   help="Path to a local training CSV file")
    p.add_argument("--val_file", default=None, required=True,
                   help="Path to a local validation CSV file")

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
                   help="Learning rate (default: 1e-5)")
    p.add_argument("--head_lr", type=float, default=5e-5,
                   help="Learning rate (default: 5e-5)")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Warmup steps (default: 0)")
    
    # Output
    p.add_argument("--output_dir", default="./results",
                   help="Directory to save the model (default: ./results)")
    p.add_argument("--logging_dir", default="./logs",
                   help="Directory for training logs (default: ./logs)")
    p.add_argument("--logging_steps", type=int, default=50,
                   help="Log every N steps (default: 50)")
    p.add_argument("--report_to", default="none",
                   help="Tracking backend: none | wandb | (default: none)")
    
    # Wandb setting
    p.add_argument("--run_name", default=None,
                   help="Experiment/run name for the logger (e.g., W&B)")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project name (optional; or set WANDB_PROJECT env var)")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity/team (optional; or set WANDB_ENTITY env var)")

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


class LRCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.optimizer = kwargs.get("optimizer", None)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            optimizer = kwargs.get("optimizer")
            if optimizer:
                logs["lr/backbone"] = optimizer.param_groups[0]["lr"]
                logs["lr/head"]     = optimizer.param_groups[1]["lr"]


def main():
    args = parse_args()
    if args.report_to == "wandb":
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Labels     : {args.num_labels}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*60}\n")

    print(f"Loading local CSV files: {args.train_file} / {args.val_file}")
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    dataset = load_dataset("csv", data_files=data_files)
    print(f"Splits available: {list(dataset.keys())}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_split = TextDataset(dataset["train"], tokenizer, args.text_col, args.label_col, args.max_length)
    eval_split  = TextDataset(dataset["validation"], tokenizer, args.text_col, args.label_col, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.num_labels,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}
    
    optimizer = torch.optim.AdamW([
        {"params": model.base_model.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ])

    num_training_steps = math.ceil(args.epochs * len(train_split) / args.train_batch_size)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),
        run_name=args.run_name,
        report_to=args.report_to,
    )

    callbacks: list[TrainerCallback] = [LRCallback()]
    if args.report_to == "wandb":
        callbacks.append(WandbCallback())
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
        data_collator=data_collator,
        callbacks=callbacks
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")

    metrics = trainer.evaluate()
    print("\nFinal eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
