import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

DATA_DIR = "data"
FEATURES_DIR = "features"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
VECTORIZER_PATH = os.path.join(FEATURES_DIR, "tfidf_vectorizer.pkl")
COMPARISON_CSV = os.path.join(FEATURES_DIR, "feature_comparison.csv")

# Keyword list for spam indicator features
SPAM_KEYWORDS = ["free", "win", "winner", "prize", "urgent", "claim", "cash", "award", "txt", "call", "guaranteed", "offer"]

def extract_handcrafted_features(raw_series: pd.Series) -> np.ndarray:
    """
    Extracts handcrafted structural and semantic features from raw_text:
    1. link count (URLs)
    2. exclamation count ('!')
    3. uppercase ratio (uppercase characters / total length)
    4. digit ratio (digits / total length)
    5. text length (character count)
    6. spam keyword match count
    """
    features = []
    url_pattern = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|co|uk)\b", re.IGNORECASE)
    kw_pattern = re.compile(r"\b(" + "|".join(SPAM_KEYWORDS) + r")\b", re.IGNORECASE)

    for text in raw_series:
        s = str(text) if text is not None else ""
        length = len(s)
        
        # Link count
        link_count = len(url_pattern.findall(s))
        
        # Exclamation count
        excl_count = s.count("!")
        
        # Uppercase ratio
        upper_count = sum(1 for c in s if c.isupper())
        upper_ratio = upper_count / length if length > 0 else 0.0
        
        # Digit ratio
        digit_count = sum(1 for c in s if c.isdigit())
        digit_ratio = digit_count / length if length > 0 else 0.0
        
        # Spam keywords count
        kw_matches = len(kw_pattern.findall(s))

        features.append([link_count, excl_count, upper_ratio, digit_ratio, length, kw_matches])

    return np.array(features, dtype=np.float64)

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        print(f"Error: {TRAIN_CSV} or {VAL_CSV} not found! Please run 'python preprocessing/split.py' first.")
        sys.exit(1)

    print("Loading train and validation datasets...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    # 1. TF-IDF Feature Extraction (unigrams + bigrams, min_df=2, max_features=50000)
    # Crucial Rule: Fit vectorizer on TRAIN only to prevent data leakage!
    print("Fitting TF-IDF Vectorizer on train['text'] only...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(train_df["text"].fillna(""))
    X_val_tfidf = tfidf.transform(val_df["text"].fillna(""))
    
    # Save TF-IDF Vectorizer
    joblib.dump(tfidf, VECTORIZER_PATH)
    print(f"Saved TF-IDF vectorizer ({X_train_tfidf.shape[1]} features) to {VECTORIZER_PATH}")

    # 2. Handcrafted Features Extraction from raw_text
    print("Extracting handcrafted features from raw_text...")
    train_handcrafted = extract_handcrafted_features(train_df["raw_text"])
    val_handcrafted = extract_handcrafted_features(val_df["raw_text"])

    scaler = StandardScaler()
    train_handcrafted_scaled = csr_matrix(scaler.fit_transform(train_handcrafted))
    val_handcrafted_scaled = csr_matrix(scaler.transform(val_handcrafted))

    # Combine TF-IDF with handcrafted features
    X_train_combined = hstack([X_train_tfidf, train_handcrafted_scaled])
    X_val_combined = hstack([X_val_tfidf, val_handcrafted_scaled])

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # 3. Model Comparison on Validation Set
    print("\nTraining baseline Logistic Regression to compare feature representations on Validation set...")
    results = []

    # Experiment A: TF-IDF Alone
    clf_tfidf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf_tfidf.fit(X_train_tfidf, y_train)
    y_val_pred_tfidf = clf_tfidf.predict(X_val_tfidf)

    results.append({
        "Feature Set": "TF-IDF Only",
        "Feature Count": X_train_tfidf.shape[1],
        "Accuracy": accuracy_score(y_val, y_val_pred_tfidf),
        "Precision": precision_score(y_val, y_val_pred_tfidf, zero_division=0),
        "Recall": recall_score(y_val, y_val_pred_tfidf, zero_division=0),
        "Validation F1": f1_score(y_val, y_val_pred_tfidf, zero_division=0)
    })

    # Experiment B: TF-IDF + Handcrafted Features
    clf_comb = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf_comb.fit(X_train_combined, y_train)
    y_val_pred_comb = clf_comb.predict(X_val_combined)

    results.append({
        "Feature Set": "TF-IDF + Handcrafted",
        "Feature Count": X_train_combined.shape[1],
        "Accuracy": accuracy_score(y_val, y_val_pred_comb),
        "Precision": precision_score(y_val, y_val_pred_comb, zero_division=0),
        "Recall": recall_score(y_val, y_val_pred_comb, zero_division=0),
        "Validation F1": f1_score(y_val, y_val_pred_comb, zero_division=0)
    })

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    print("\n=== Feature Representation Comparison on Validation Set ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved feature comparison results to {COMPARISON_CSV}")

if __name__ == "__main__":
    main()
