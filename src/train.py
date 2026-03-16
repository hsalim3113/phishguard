"""
train.py — Training pipeline for PhishGuard.

This script runs the full training pipeline from preprocessed data to a
saved model. The steps are:
  1. Load the cleaned dataset from data/processed/processed.csv
  2. Drop any remaining NaN rows and force string types on the text column
  3. Split into 80/20 train/test sets using a stratified split
  4. Vectorise the text with TF-IDF (bigrams, 30k features)
  5. Train a Logistic Regression classifier
  6. Evaluate on the held-out test set and print all key metrics
  7. Save the model and vectoriser to models/
  8. Write the evaluation report and a metrics JSON to outputs/evaluation/
     so the Streamlit app can load the numbers without needing to retrain
"""

import json
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = Path("data/processed/processed.csv")
MODEL_DIR = Path("models")
EVAL_DIR = Path("outputs/evaluation")

MODEL_PATH = MODEL_DIR / "model_logreg.joblib"
VEC_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
REPORT_PATH = EVAL_DIR / "evaluation_report.txt"
METRICS_PATH = EVAL_DIR / "metrics.json"


def main():
    df = pd.read_csv(DATA_PATH)

    # Some rows end up with missing text after combining the three datasets —
    # dropping them here rather than letting the vectoriser silently choke on NaN
    df = df.dropna(subset=["text_combined"])

    X = df["text_combined"]
    y = df["label"].astype(int)

    # Pandas can read mixed-type columns as float (e.g. "nan" cells), so we
    # force everything to string to avoid a type error inside the vectoriser
    X = X.astype(str)

    # Fixing the random seed means the train/test split is identical on every
    # run, so evaluation numbers are directly comparable if I retrain later
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # max_features=30000 caps vocabulary size to keep memory usage reasonable;
    # ngram_range=(1,2) adds bigrams so the model can pick up two-word patterns
    # like "click here" or "verify account" that a unigram model would miss
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english")

    # fit_transform on training data only — fitting on the test set too
    # would be data leakage and would inflate the evaluation scores
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # I considered SVM and Random Forest but chose Logistic Regression for
    # three main reasons:
    #   1. The coefficients map directly to word importance, which pairs well
    #      with TF-IDF and makes the model easier to explain in the write-up
    #   2. It trains quickly on large sparse matrices — important when I was
    #      iterating on the dataset and retraining frequently during development
    #   3. It gives properly calibrated probabilities rather than raw scores,
    #      which means the confidence percentage shown in the UI actually means
    #      something rather than being an arbitrary number
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    # predict_proba returns [P(legitimate), P(phishing)] for each sample —
    # we only need the phishing probability (class 1) for the ROC-AUC score
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    # --- Evaluation metrics ---
    # Computing each one separately so they can be printed clearly and
    # saved to JSON for the Streamlit sidebar
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n===== Model Evaluation =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # --- Confusion matrix ---
    # Layout is:  [TN  FP]
    #             [FN  TP]
    # False negatives (phishing labelled as legitimate) are the most
    # dangerous error for this use case, so it's worth checking FN directly
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix (rows=Actual, cols=Predicted) ---")
    print(f"                Predicted Legitimate  Predicted Phishing")
    print(f"Actual Legitimate      {cm[0][0]:<20} {cm[0][1]}")
    print(f"Actual Phishing        {cm[1][0]:<20} {cm[1][1]}")

    # Full sklearn report for completeness — includes per-class breakdown
    report = classification_report(y_test, y_pred, digits=3)
    print("\n--- Full Classification Report ---")
    print(report)

    # --- Persist evaluation outputs ---
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Write the full report to a file so I can review it later without
    # having to retrain — useful when writing up the results section
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("===== PhishGuard Model Evaluation Report =====\n\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix (rows=Actual, cols=Predicted):\n")
        f.write(str(cm) + "\n\n")
        f.write("Full Classification Report:\n")
        f.write(report)
    print(f"Saved evaluation report: {REPORT_PATH}")

    # Dump the key metrics to JSON so the Streamlit sidebar can load and
    # display them without needing to re-run training each time the app starts
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {METRICS_PATH}")

    # --- Save model and vectoriser ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    dump(vec, VEC_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VEC_PATH}")


if __name__ == "__main__":
    main()
