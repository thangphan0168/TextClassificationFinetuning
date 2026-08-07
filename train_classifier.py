"""
Fine-tune a text classifier with HuggingFace Transformers using Stratified K-Fold.

Usage
-----
# Standard Train/Val Split (Weighted Loss enabled)
python train_classifier.py \
    --train_file train.csv \
    --val_file val.csv \
    --use_weighted_loss \
    --weight_power 0.5 \
    --text_col review \
    --num_labels 5 \
    --metric_for_best_model f1

# 5-Fold Stratified Cross-Validation (Uses ONLY train_file)
python train_classifier.py \
    --train_file full_dataset.csv \
    --k_folds 5 \
    --use_weighted_loss \
    --text_col review \
    --num_labels 5 \
    --metric_for_best_model f1
"""

import argparse
import os
import gc

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
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
                   help="Path to a local training file (or full dataset if doing k-fold)")
    p.add_argument("--val_file", default=None, required=False,
                   help="Path to a local validation file (Ignored if k_folds > 1)")
    p.add_argument("--file_type", default="csv", required=False,
                   help="File type: csv | parquet (default: csv)")
                   
    # K-Fold Argument
    p.add_argument("--k_folds", type=int, default=1,
                   help="Number of folds for stratified cross-validation (default: 1, standard train/val split)")

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
    p.add_argument("--use_weighted_loss", action="store_true",
                   help="Enable dynamically calculated class weights for CrossEntropy loss")
    p.add_argument("--weight_power", type=float, default=1.0,
                   help="Power parameter for class weights (default: 1.0)")
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
    p.add_argument("--head_lr_multiplier", type=float, default=None,
                   help="Multiplier for backbone_lr to get head_lr. Overrides head_lr if passed.")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Warmup steps (default: 0)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze all backbone weights and train only the classification head")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducibility (default: 0)")

    # Output and logging
    p.add_argument("--output_dir", default="./results",
                   help="Directory to save the model (default: ./results)")
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


def get_weighted_loss_func(class_weights):
    def custom_loss_func(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits
        weights = class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss
    return custom_loss_func


def get_class_weights(labels, power=1.0):
    unique_classes = np.unique(labels)
    weights_array = compute_class_weight(
        class_weight="balanced", 
        classes=unique_classes, 
        y=labels
    )
    if power != 1.0:
        weights_array = weights_array ** power
    return torch.tensor(weights_array, dtype=torch.float)


def freeze_backbone(model) -> int:
    """Freeze all backbone parameters and return the count of frozen params."""
    frozen = 0
    for param in model.base_model.parameters():
        param.requires_grad = False
        frozen += param.numel()
    return frozen


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)        
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_evaluate_split(args, train_split, eval_split, train_labels, tokenizer, data_collator, fold=None):
    """
    Handles the initialization, training, and evaluation loop. 
    Extracted so it can be re-run freshly for every fold.
    """
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    out_dir = f"{args.output_dir}{fold_suffix}"
    current_run_name = f"{args.run_name}{fold_suffix}" if args.run_name else None

    if args.report_to == "wandb" and args.wandb_project:
        wandb.init(project=args.wandb_project, name=current_run_name, config=vars(args), reinit=True)

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model,
    #     num_labels=args.num_labels,
    # )
    config = AutoConfig.from_pretrained(args.model, num_labels=args.num_labels)
    if args.freeze_backbone:
        config.dropout = 0.0
        config.attention_dropout = 0.0
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels, config=config)

    loss_func = None
    if args.use_weighted_loss:
        class_weights_tensor = get_class_weights(train_labels, power=args.weight_power)
        if fold is None or fold == 1:
            print(f"Using Weighted Loss (power={args.weight_power}): {class_weights_tensor}")
        loss_func = get_weighted_loss_func(class_weights_tensor)

    if args.freeze_backbone:
        frozen_count = freeze_backbone(model)
        if fold is None or fold == 1:
            print(f"Backbone frozen: {frozen_count:,} parameters will not be updated.\n")
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.head_lr)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": args.head_lr},
            {"params": model.base_model.parameters(), "lr": args.backbone_lr},
        ])

    training_args = TrainingArguments(
        output_dir=out_dir,
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
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),
        run_name=current_run_name,
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
        compute_loss_func=loss_func,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    metrics = trainer.evaluate()
    if args.report_to == "wandb":
        wandb.finish()
        
    del model, trainer, optimizer, loss_func
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.head_lr_multiplier is not None:
        args.head_lr = args.backbone_lr * args.head_lr_multiplier

    if args.k_folds <= 1 and not args.val_file:
        raise ValueError("You must provide --val_file if --k_folds is set to 1.")

    if args.report_to == "wandb" and args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Labels     : {args.num_labels}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  K-Folds    : {args.k_folds}")
    print(f"  Use weights: {args.use_weighted_loss} (power: {args.weight_power})")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Best metric: {args.metric_for_best_model}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*60}\n")

    # Load Data
    data_files = {"train": args.train_file}
    if args.val_file and args.k_folds <= 1:
        data_files["validation"] = args.val_file
        
    print(f"Loading files: {data_files}")
    dataset = load_dataset(args.file_type, data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.k_folds > 1:
        print(f"Initializing {args.k_folds}-Fold Stratified Cross Validation...\n")
        full_data = dataset["train"]
        labels = full_data[args.label_col]
        
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            print(f"\n{'*'*40}")
            print(f"   STARTING FOLD {fold}/{args.k_folds}")
            print(f"{'*'*40}")
            
            train_subset = full_data.select(train_idx)
            val_subset = full_data.select(val_idx)
            fold_train_labels = train_subset[args.label_col]
            
            train_split = TextDataset(train_subset, tokenizer, args.text_col, args.label_col, args.max_length)
            eval_split  = TextDataset(val_subset, tokenizer, args.text_col, args.label_col, args.max_length)
            
            metrics = train_evaluate_split(
                args, train_split, eval_split, fold_train_labels, tokenizer, data_collator, fold=fold
            )
            fold_metrics.append(metrics)
            
            print(f"\nFold {fold} Results:")
            for k, v in metrics.items():
                if k.startswith("eval_"):
                    print(f"  {k}: {v:.4f}")

        # Aggregate results
        print(f"\n{'='*60}")
        print(f"FINAL AGGREGATED METRICS ACROSS {args.k_folds} FOLDS:")
        print(f"{'='*60}")
        metric_keys = [k for k in fold_metrics[0].keys() if k.startswith("eval_")]
        
        for key in metric_keys:
            values = [m[key] for m in fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key}: {mean_val:.4f} (± {std_val:.4f})")

    else:
        print("Running standard single Train/Validation split...\n")
        train_labels = dataset["train"][args.label_col]
        train_split = TextDataset(dataset["train"], tokenizer, args.text_col, args.label_col, args.max_length)
        eval_split  = TextDataset(dataset["validation"], tokenizer, args.text_col, args.label_col, args.max_length)
        
        metrics = train_evaluate_split(
            args, train_split, eval_split, train_labels, tokenizer, data_collator, fold=None
        )
        
        print("\nFinal eval metrics:")
        for k, v in metrics.items():
            if k.startswith("eval_"):
                print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
