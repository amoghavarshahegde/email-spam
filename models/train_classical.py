import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "data"
MODELS_DIR = "models"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
RESULTS_CSV = os.path.join(MODELS_DIR, "classical_results.csv")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        print(f"Error: {TRAIN_CSV} or {VAL_CSV} not found! Please run 'python preprocessing/split.py' first.")
        sys.exit(1)

    print("Loading train and validation datasets...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"].values
    X_val = val_df["text"].fillna("")
    y_val = val_df["label"].values

    print(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples.")

    # Base TF-IDF configuration matching Phase 3 specs
    tfidf_params = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 50000,
        "sublinear_tf": True
    }

    # Model Pipelines
    models = {
        "NaiveBayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB(alpha=0.2))
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42, max_iter=2000))
        ])
    }

    results = []
    trained_pipelines = {}

    print("\n--- Training & Tuning Classical Models ---")

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        if name == "LogisticRegression":
            # 5-fold GridSearchCV tuning on train only, optimizing for F1
            param_grid = {
                "clf__C": [0.1, 0.5, 1.0, 5.0, 10.0]
            }
            grid = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="f1",
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_
            print(f"  Best params for {name}: {grid.best_params_} (Mean CV F1: {grid.best_score_:.4f})")
            model_to_eval = best_pipeline
        else:
            pipeline.fit(X_train, y_train)
            model_to_eval = pipeline

        # Evaluate on validation set
        val_preds = model_to_eval.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        f1 = f1_score(y_val, val_preds, zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        # Save model pipeline with joblib
        model_file = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model_to_eval, model_file)
        trained_pipelines[name] = model_to_eval
        print(f"  Saved pipeline to {model_file}")
        print(f"  Val Results -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "="*50)
    print("=== Validation Set Classical Model Results ===")
    print("="*50)
    print(results_df.to_string(index=False))
    print(f"\nSaved results to {RESULTS_CSV}")

    # Determine and print the best model
    best_row = results_df.loc[results_df["F1"].idxmax()]
    print("\n" + "*"*50)
    print(f" BEST MODEL: {best_row['Model']} with Validation F1: {best_row['F1']:.4f}")
    print("*"*50 + "\n")

if __name__ == "__main__":
    main()
