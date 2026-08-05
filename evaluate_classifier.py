"""
Evaluate a fine-tuned HuggingFace text classification model.

Usage
-----
python evaluate_classifier.py \
    --model_path ./results_fold_1 \
    --test_file test.csv \
    --text_col review \
    --label_col label \
    --save_predictions predictions.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a fine-tuned HuggingFace model on a test set."
    )

    # Model and Data
    p.add_argument("--model_path", required=True,
                   help="Path to the saved model checkpoint (e.g., ./results)")
    p.add_argument("--test_file", required=True,
                   help="Path to the local test file (CSV or Parquet)")
    p.add_argument("--file_type", default="csv",
                   help="File type: csv | parquet (default: csv)")

    # Columns
    p.add_argument("--text_col", default="text",
                   help="Name of the text column in your dataset (default: text)")
    p.add_argument("--label_col", default="label",
                   help="Name of the label column in your dataset (default: label)")

    # Settings
    p.add_argument("--max_length", type=int, default=512,
                   help="Max token length (default: 512)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Per-device eval batch size (default: 16)")
    
    # Outputs
    p.add_argument("--save_predictions", default=None,
                   help="Optional: Path to save a CSV of the raw data + model predictions (e.g., predictions.csv)")

    return p.parse_args()


class TextDataset(torch.utils.data.Dataset):
    """Same dataset wrapper used during training."""
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
        # Ensure labels are explicitly passed to compute loss / metrics later
        if self.label_col in item:
            encoding["labels"] = item[self.label_col]
        return encoding


def compute_metrics(eval_pred):
    """Computes same metrics as the training script."""
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


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")

    print(f"\n{'='*60}")
    print(f"  Model Path : {args.model_path}")
    print(f"  Test File  : {args.test_file}")
    print(f"  Batch Size : {args.batch_size}")
    print(f"{'='*60}\n")

    # 1. Load Data
    print("Loading dataset...")
    dataset = load_dataset(args.file_type, data_files={"test": args.test_file})
    test_data = dataset["test"]

    # 2. Load Model and Tokenizer
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    # 3. Prepare Dataset for Trainer
    test_split = TextDataset(test_data, tokenizer, args.text_col, args.label_col, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Setup dummy TrainingArguments for Evaluation
    # (Trainer requires TrainingArguments even if we only evaluate)
    eval_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=args.batch_size,
        fp16=torch.cuda.is_available(),
        report_to="none", # Disable wandb logging for this script
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # 5. Run Prediction / Evaluation
    print("\nRunning evaluation...")
    predictions = trainer.predict(test_split)

    # 6. Print Metrics
    print(f"\n{'='*60}")
    print("FINAL TEST METRICS:")
    print(f"{'='*60}")
    for k, v in predictions.metrics.items():
        # trainer.predict automatically prepends 'test_' to metric names
        clean_key = k.replace("test_", "")
        print(f"  {clean_key.capitalize():<12}: {v:.4f}" if isinstance(v, float) else f"  {clean_key:<12}: {v}")

    # 7. Save Predictions (Optional)
    if args.save_predictions:
        print(f"\nSaving predictions to {args.save_predictions}...")
        # Get raw text and true labels
        texts = test_data[args.text_col]
        true_labels = test_data[args.label_col]
        
        # Get predicted labels (argmax of logits)
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        # Create a DataFrame and save
        df = pd.DataFrame({
            args.text_col: texts,
            "true_label": true_labels,
            "predicted_label": pred_labels
        })
        
        # Add probability scores (softmax) if needed
        import torch.nn.functional as F
        logits_tensor = torch.tensor(predictions.predictions)
        probs = F.softmax(logits_tensor, dim=-1).numpy()
        
        # Save highest probability confidence score
        df["confidence_score"] = np.max(probs, axis=-1)
        
        df.to_csv(args.save_predictions, index=False)
        print("Done!")

if __name__ == "__main__":
    main()
