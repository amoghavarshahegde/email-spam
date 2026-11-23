# SMS / Email Spam Detection

A complete machine-learning pipeline that classifies SMS and email messages as **spam** or **ham** (not spam).  
Built with Python, scikit-learn, and Streamlit.

---

## Project Overview

| Metric | Value |
|--------|-------|
| Dataset | UCI SMS Spam Collection (~5,500 messages) |
| Class balance | ~87 % ham, 13 % spam |
| Best model (test F1) | LinearSVC – **0.9362** |
| AUC (all models) | > 0.98 |

---

## Folder Ownership

| Folder | Owner / Team | Contents |
|--------|--------------|----------|
| `data/` | Data team | Raw CSV, cleaned CSV, train/val/test splits, EDA charts |
| `preprocessing/` | Preprocessing team | Text cleaning, train-val-test splitting |
| `features/` | Features team | TF-IDF vectorizer, handcrafted feature comparison |
| `models/` | Models team | Classical sklearn pipelines, DistilBERT fine-tuning script |
| `evaluation/` | Evaluation team | Final metrics, confusion matrices, ROC curve, misclassified examples |
| `app/` | App team | Streamlit web app |

---

## Run Order

```bash
# 1. Data preparation (download + clean)
python data/prepare_data.py

# 2. Exploratory Data Analysis  (saves charts to data/)
python data/eda.py

# 3. Text cleaning + train/val/test split
python preprocessing/split.py

# 4. TF-IDF + handcrafted feature comparison
python features/build_features.py

# 5. Train classical models (Naive Bayes, Logistic Regression, LinearSVC)
python models/train_classical.py

# 6. Evaluate all models on the held-out test set
python evaluation/evaluate.py

# 7. Launch the Streamlit web app
streamlit run app/app.py
```

> **Deep learning (DistilBERT)** – Run on Google Colab (free T4 GPU).  
> See the header of `models/train_distilbert.py` for step-by-step Colab instructions.

---

## Setup

```bash
pip install -r requirements.txt
```

**requirements.txt** installs:  
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`, `nltk`, `transformers`, `datasets`, `torch`

---

## Key Design Decisions

- **No data leakage**: TF-IDF vectorizer and scalers are **fitted on train only**, then applied to val/test.
- **Stratified splits**: 70 / 15 / 15 with `random_state=42` preserves the spam/ham ratio in every split.
- **False Positives are highest priority**: Blocking a real email is worse than missing spam.  
  NaiveBayes has the fewest FP (1) but most FN (11); LinearSVC/LR balance both.
- **Raw text for DistilBERT**: The transformer uses un-cleaned text — it benefits from original casing and punctuation.

---

## Output Files

| File | Description |
|------|-------------|
| `data/emails_raw.csv` | Cleaned, deduplicated full dataset |
| `data/train.csv`, `val.csv`, `test.csv` | Stratified splits |
| `data/*.png` | EDA charts |
| `features/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `features/feature_comparison.csv` | TF-IDF vs TF-IDF+Handcrafted comparison |
| `models/*.pkl` | Trained sklearn pipelines |
| `models/classical_results.csv` | Validation-set metrics |
| `evaluation/final_results.csv` | Test-set metrics for all models |
| `evaluation/confusion_*.png` | Confusion matrices |
| `evaluation/roc_curve.png` | ROC curves |
| `evaluation/misclassified.csv` | All misclassified test examples |
