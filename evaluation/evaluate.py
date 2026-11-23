"""
evaluate.py – Final evaluation on the held-out test set.
OWNER: evaluation/ team

Evaluates every trained .pkl model found in models/ against data/test.csv.
Produces:
  evaluation/final_results.csv       – accuracy / precision / recall / F1
  evaluation/confusion_<model>.png   – confusion matrix per model
  evaluation/roc_curve.png           – ROC curves (all models on one plot)
  evaluation/misclassified.csv       – false positives + false negatives
Run AFTER train_classical.py has saved .pkl files.
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works without a display
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

DATA_DIR  = "data"
MODEL_DIR = "models"
EVAL_DIR  = "evaluation"
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ── Guard: test set must exist ──────────────────────────────────────────
    if not os.path.exists(TEST_CSV):
        print(f"Error: {TEST_CSV} not found! Run 'python preprocessing/split.py' first.")
        sys.exit(1)

    print(f"Loading test set from {TEST_CSV} ...")
    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df["text"].fillna("").tolist()
    y_test = test_df["label"].values

    print(f"Test set size: {len(test_df)}  "
          f"(Ham: {(y_test==0).sum()}, Spam: {(y_test==1).sum()})")

    # ── Load all trained .pkl pipelines ────────────────────────────────────
    model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl")))
    if not model_paths:
        print(f"Error: No .pkl files found in {MODEL_DIR}/! "
              "Run 'python models/train_classical.py' first.")
        sys.exit(1)

    print(f"\nFound {len(model_paths)} model(s): "
          f"{[os.path.basename(p) for p in model_paths]}\n")

    all_results = []
    roc_data    = {}   # {model_name: (fpr, tpr, auc_val)}
    misclassified_rows = []

    for path in model_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"=== {name} ===")
        model = joblib.load(path)

        # Predict labels and probabilities (if available)
        y_pred = model.predict(X_test)

        # Try to get probability scores for ROC (some models don't support it)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            has_proba = True
        except AttributeError:
            # LinearSVC doesn't have predict_proba; use decision_function instead
            try:
                score = model.decision_function(X_test)
                # Normalise to [0,1] range so it looks like a probability
                y_prob = (score - score.min()) / (score.max() - score.min() + 1e-9)
                has_proba = True
            except AttributeError:
                has_proba = False
                y_prob = None

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")

        all_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
        })

        # ── Confusion Matrix ────────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Ham (0)", "Spam (1)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix – {name}")
        plt.tight_layout()
        cm_path = os.path.join(EVAL_DIR, f"confusion_{name}.png")
        fig.savefig(cm_path, dpi=200)
        plt.close()
        print(f"  Confusion matrix saved -> {cm_path}")

        # Collect false positives (ham predicted as spam – VERY BAD)
        # and false negatives (spam predicted as ham)
        fp_mask = (y_test == 0) & (y_pred == 1)
        fn_mask = (y_test == 1) & (y_pred == 0)
        for idx in np.where(fp_mask)[0]:
            misclassified_rows.append({
                "model": name,
                "error_type": "FALSE_POSITIVE (ham->spam)",
                "true_label": 0,
                "pred_label": 1,
                "text": test_df.iloc[idx]["raw_text"] if "raw_text" in test_df.columns else X_test[idx],
            })
        for idx in np.where(fn_mask)[0]:
            misclassified_rows.append({
                "model": name,
                "error_type": "false_negative (spam→ham)",
                "true_label": 1,
                "pred_label": 0,
                "text": test_df.iloc[idx]["raw_text"] if "raw_text" in test_df.columns else X_test[idx],
            })

        fp_count = fp_mask.sum()
        fn_count = fn_mask.sum()
        print(f"  !! False Positives (real email blocked!): {fp_count}")
        print(f"     False Negatives (spam let through)   : {fn_count}")

        # ROC data
        if has_proba and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
            auc_val = auc(fpr, tpr)
            roc_data[name] = (fpr, tpr, auc_val)
            print(f"  AUC      : {auc_val:.4f}")
        print()

    # ── ROC Curve (all models) ───────────────────────────────────────────────
    if roc_data:
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#2b8a3e", "#c92a2a", "#1864ab", "#862e9c"]
        for i, (mname, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
            ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                    label=f"{mname} (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves – All Models on Test Set")
        ax.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(EVAL_DIR, "roc_curve.png")
        fig.savefig(roc_path, dpi=200)
        plt.close()
        print(f"ROC curve saved -> {roc_path}")

    # ── Save results CSV ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(EVAL_DIR, "final_results.csv"), index=False)
    print("\n=== FINAL TEST-SET RESULTS ===")
    print(results_df.to_string(index=False))

    # ── Save misclassified ──────────────────────────────────────────────────
    if misclassified_rows:
        misc_df = pd.DataFrame(misclassified_rows)
        misc_path = os.path.join(EVAL_DIR, "misclassified.csv")
        misc_df.to_csv(misc_path, index=False)
        print(f"\nMisclassified examples saved -> {misc_path}")
        fp_df = misc_df[misc_df["error_type"].str.startswith("FALSE_POSITIVE")]
        if not fp_df.empty:
            print(f"\n!! FALSE POSITIVES (legitimate email blocked as spam) - "
                  f"These are highest-priority errors:\n")
            for _, row in fp_df.drop_duplicates("text").head(5).iterrows():
                print(f"  [{row['model']}] \"{row['text'][:100]}...\"")

    best = results_df.loc[results_df["F1"].idxmax()]
    print(f"\n** Best model on TEST set: {best['Model']}  "
          f"F1={best['F1']:.4f}  Acc={best['Accuracy']:.4f}")

if __name__ == "__main__":
    main()
