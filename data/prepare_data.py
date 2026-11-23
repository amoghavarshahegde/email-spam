import os
import sys
import pandas as pd

# Paths
DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "mail_data.csv")
OUTPUT_RAW = os.path.join(DATA_DIR, "emails_raw.csv")
OUTPUT_SAMPLE = os.path.join(DATA_DIR, "sample_100.csv")

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found! Please place mail_data.csv in the data/ folder first.")
        sys.exit(1)

    print(f"Loading raw dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    # Rename columns to standard text and label
    # mail_data.csv has columns ['Category', 'Message']
    if "Category" in df.columns and "Message" in df.columns:
        df = df.rename(columns={"Category": "label", "Message": "text"})
    elif "label" not in df.columns or "text" not in df.columns:
        # Fallback to positional mapping
        df.columns = ["label", "text"] + list(df.columns[2:])

    # Map labels: spam -> 1, ham -> 0
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})

    initial_count = len(df)
    print(f"Initial row count: {initial_count}")

    # Drop NaN or missing values
    df = df.dropna(subset=["text", "label"])

    # Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Clean text whitespace and drop empty strings
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Drop duplicate messages
    df = df.drop_duplicates(subset=["text"])
    
    total_count = len(df)
    class_counts = df["label"].value_counts().to_dict()

    print("\n=== Data Cleaning Summary ===")
    print(f"Dropped {initial_count - total_count} invalid / duplicate rows.")
    print(f"Total remaining records: {total_count}")
    print(f"Class counts -> Ham (0): {class_counts.get(0, 0)}, Spam (1): {class_counts.get(1, 0)}")
    ham_pct = (class_counts.get(0, 0) / total_count) * 100
    spam_pct = (class_counts.get(1, 0) / total_count) * 100
    print(f"Class proportions -> Ham: {ham_pct:.2f}%, Spam: {spam_pct:.2f}%")

    # Save cleaned raw data
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"Saved full clean dataset to {OUTPUT_RAW}")

    # Save a 100-row sample for quick inspection
    sample_df = df.sample(n=min(100, len(df)), random_state=42)
    sample_df.to_csv(OUTPUT_SAMPLE, index=False)
    print(f"Saved 100-row sample to {OUTPUT_SAMPLE}")

if __name__ == "__main__":
    main()
