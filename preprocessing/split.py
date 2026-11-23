import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add current directory and preprocessing to sys.path to enable clean imports
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("preprocessing"))

from cleaning import clean_text

DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "emails_raw.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found! Run 'python data/prepare_data.py' first.")
        sys.exit(1)

    print(f"Loading cleaned raw dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Preserve raw text
    df["raw_text"] = df["text"].astype(str)

    print("Applying text cleaning (lowercasing, entity masking, symbol removal)...")
    df["text"] = df["raw_text"].apply(clean_text)

    # Drop any rows that became completely empty after cleaning
    df = df[df["text"].str.strip() != ""]

    # Stratified 70% train, 15% validation, 15% test split (random_state=42)
    # Step 1: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    # Step 2: Split 30% temp equally into 15% val and 15% test (0.50 of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    # Ensure consistent column ordering: raw_text, text, label
    cols = ["raw_text", "text", "label"]
    train_df = train_df[cols]
    val_df = val_df[cols]
    test_df = test_df[cols]

    # Save to data/
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("\n=== Dataset Split Summary (Stratified 70/15/15, Seed 42) ===")
    for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        total = len(split_df)
        spam_count = (split_df["label"] == 1).sum()
        ham_count = (split_df["label"] == 0).sum()
        spam_ratio = (spam_count / total) * 100 if total > 0 else 0
        print(f"{name:12} -> Total: {total:4} | Ham: {ham_count:4} | Spam: {spam_count:4} | Spam Ratio: {spam_ratio:.2f}%")

    print(f"\nSaved splits:\n - {TRAIN_CSV}\n - {VAL_CSV}\n - {TEST_CSV}")

if __name__ == "__main__":
    main()
