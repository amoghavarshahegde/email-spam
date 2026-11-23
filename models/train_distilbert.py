"""
==============================================================================
 train_distilbert.py  –  Fine-tune DistilBERT for Spam Detection
==============================================================================
 OWNER: models/ team
 This script is designed to run on Google Colab with a FREE T4 GPU.
 DO NOT run full training locally (it would take many hours on CPU).

 ── COLAB QUICK-START ────────────────────────────────────────────────────────
 1. Open https://colab.research.google.com/
 2. Runtime → Change runtime type → GPU (T4)
 3. Upload this file OR paste its contents into a Colab cell.
 4. Run the installation cell first:
      !pip install transformers datasets scikit-learn pandas torch
 5. Upload data/train.csv and data/val.csv from your local machine:
      from google.colab import files
      uploaded = files.upload()   # select train.csv and val.csv
 6. Run this script normally:
      !python train_distilbert.py
    OR for a quick 2000-row test run:
      !python train_distilbert.py --quick-test
 7. After training, download the saved model folder:
      from google.colab import files
      import shutil
      shutil.make_archive('distilbert_spam', 'zip', 'distilbert_spam/')
      files.download('distilbert_spam.zip')
 8. Unzip distilbert_spam.zip into the repo root so evaluate.py can find it.
 ──────────────────────────────────────────────────────────────────────────────
 EXPECTED RUNTIME (Colab T4):
   Full dataset (~3600 rows, 2 epochs, batch 16):  ~8 minutes
   Quick-test   (2000 rows, 2 epochs, batch 16):   ~5 minutes
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for spam detection")
parser.add_argument(
    "--quick-test",
    action="store_true",
    help="Use only 2000 rows of training data for a fast smoke-test run"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Imports  (heavy – only needed at runtime)
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install transformers datasets scikit-learn torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 128
EPOCHS       = 2
BATCH_SIZE   = 16
RANDOM_SEED  = 42
OUTPUT_DIR   = "distilbert_spam"       # saved model directory

TRAIN_CSV = "data/train.csv"
VAL_CSV   = "data/val.csv"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
    print(f"Error: {TRAIN_CSV} or {VAL_CSV} not found!")
    print("Make sure you have uploaded train.csv and val.csv to the working directory.")
    sys.exit(1)

print("Loading dataset splits...")
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

# Use raw_text (un-cleaned) for the transformer – it benefits from original casing/punctuation
train_df = train_df[["raw_text", "label"]].rename(columns={"raw_text": "text"})
val_df   = val_df[["raw_text", "label"]].rename(columns={"raw_text": "text"})

# Quick-test mode: cap at 2000 training rows and 400 validation rows
if args.quick_test:
    print(f"[QUICK-TEST MODE] Sampling 2000 train / 400 val rows ...")
    train_df = train_df.sample(n=min(2000, len(train_df)), random_state=RANDOM_SEED)
    val_df   = val_df.sample(n=min(400,  len(val_df)),   random_state=RANDOM_SEED)

print(f"Train rows: {len(train_df)}  |  Val rows: {len(val_df)}")

# ---------------------------------------------------------------------------
# Tokenise
# ---------------------------------------------------------------------------
print(f"Loading tokenizer: {MODEL_NAME} ...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    """Tokenize a batch of raw text messages."""
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# Convert to Hugging Face Dataset objects
train_hf = Dataset.from_pandas(train_df.reset_index(drop=True))
val_hf   = Dataset.from_pandas(val_df.reset_index(drop=True))

print("Tokenizing ...")
train_hf = train_hf.map(tokenize, batched=True, batch_size=512)
val_hf   = val_hf.map(tokenize,   batched=True, batch_size=512)

# Rename 'label' to match Trainer's expected column name
train_hf = train_hf.rename_column("label", "labels")
val_hf   = val_hf.rename_column("label", "labels")

# Keep only the columns the model needs
train_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_hf.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
print(f"Loading model: {MODEL_NAME} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,          # 0=ham, 1=spam
    id2label={0: "ham", 1: "spam"},
    label2id={"ham": 0, "spam": 1}
)

# ---------------------------------------------------------------------------
# Metrics function
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall and F1 during evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall":    recall_score(labels, predictions, zero_division=0),
        "f1":        f1_score(labels, predictions, zero_division=0),
    }

# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",          # evaluate after every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=RANDOM_SEED,
    logging_steps=50,
    report_to="none",               # disable wandb / tensorboard
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print(f"\nStarting fine-tuning ({EPOCHS} epochs, batch size {BATCH_SIZE}) ...")
trainer.train()

# ---------------------------------------------------------------------------
# Save model + tokenizer
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel and tokenizer saved to '{OUTPUT_DIR}/'")

# ---------------------------------------------------------------------------
# Final evaluation on validation set
# ---------------------------------------------------------------------------
print("\nFinal evaluation on validation set:")
metrics = trainer.evaluate()
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\nDone! Download the 'distilbert_spam/' folder and place it in your repo root.")
print("Then run: python evaluation/evaluate.py")
