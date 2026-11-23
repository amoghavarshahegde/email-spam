import os
import sys
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "emails_raw.csv")

def get_top_words(texts, n=15):
    # Common English stop words to exclude for meaningful word frequency comparison
    STOPWORDS = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "is", "it", "this", "that", "i", "you", "he", "she", "we", "they", "me", "my",
        "your", "our", "all", "so", "be", "are", "was", "were", "have", "has", "had",
        "do", "does", "did", "but", "by", "from", "not", "no", "if", "as", "what",
        "will", "can", "just", "up", "out", "how", "when", "there", "then", "now",
        "get", "got", "its", "am", "about", "u", "ur", "ok", "im", "go", "come", "lor"
    }
    words = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        words.extend([w for w in tokens if w not in STOPWORDS])
    return Counter(words).most_common(n)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found! Please run 'python data/prepare_data.py' first.")
        sys.exit(1)

    print(f"Loading data from {INPUT_CSV} for Exploratory Data Analysis...")
    df = pd.read_csv(INPUT_CSV)

    # Compute length features
    df["char_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    print("\n=== Dataset Summary Statistics ===")
    print(f"Total messages: {len(df)}")
    print(f"Ham (0) count: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.2f}%)")
    print(f"Spam (1) count: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.2f}%)")
    
    print("\nMessage Character Length Statistics:")
    print(df.groupby("label")["char_length"].describe())

    print("\nMessage Word Count Statistics:")
    print(df.groupby("label")["word_count"].describe())

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts()
    bars = ax.bar(["Ham (0)", "Spam (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#2b8a3e", "#c92a2a"])
    ax.set_title("Class Distribution (Ham vs. Spam)")
    ax.set_ylabel("Number of Messages")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval} ({yval/len(df)*100:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    chart1_path = os.path.join(DATA_DIR, "class_distribution.png")
    fig.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"\nSaved class distribution chart to {chart1_path}")

    # 2. Message Length by Class Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[df["label"] == 0]["char_length"], bins=50, alpha=0.6, label="Ham", color="#2b8a3e", density=True)
    ax.hist(df[df["label"] == 1]["char_length"], bins=50, alpha=0.6, label="Spam", color="#c92a2a", density=True)
    ax.set_xlim(0, 300)
    ax.set_title("Message Character Length Distribution by Class")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    chart2_path = os.path.join(DATA_DIR, "message_length.png")
    fig.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"Saved message length distribution chart to {chart2_path}")

    # 3. Top 15 Words for Spam vs Ham Plot
    spam_words = get_top_words(df[df["label"] == 1]["text"], n=15)
    ham_words = get_top_words(df[df["label"] == 0]["text"], n=15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Ham Top Words
    if ham_words:
        h_words, h_counts = zip(*ham_words)
        axes[0].barh(range(len(h_words)), h_counts, color="#2b8a3e")
        axes[0].set_yticks(range(len(h_words)))
        axes[0].set_yticklabels(h_words)
        axes[0].invert_yaxis()
        axes[0].set_title("Top 15 Informative Words in Ham (Non-Spam)")
        axes[0].set_xlabel("Frequency")

    # Spam Top Words
    if spam_words:
        s_words, s_counts = zip(*spam_words)
        axes[1].barh(range(len(s_words)), s_counts, color="#c92a2a")
        axes[1].set_yticks(range(len(s_words)))
        axes[1].set_yticklabels(s_words)
        axes[1].invert_yaxis()
        axes[1].set_title("Top 15 Informative Words in Spam")
        axes[1].set_xlabel("Frequency")

    plt.tight_layout()
    chart3_path = os.path.join(DATA_DIR, "top_words.png")
    fig.savefig(chart3_path, dpi=200)
    plt.close()
    print(f"Saved top words chart to {chart3_path}")

if __name__ == "__main__":
    main()
