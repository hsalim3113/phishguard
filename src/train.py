"""
train.py — Training pipeline for PhishGuard.

This script handles the full model training process:
  1. Load the preprocessed dataset from data/processed/processed.csv
  2. Split the data into training and test sets (80/20 stratified split)
  3. Vectorise the text using TF-IDF with bigrams and a 30,000-feature cap
  4. Train a Logistic Regression classifier on the TF-IDF vectors
  5. Evaluate the model on the held-out test set (accuracy, precision, recall,
     F1-score, ROC-AUC, and confusion matrix)
  6. Save the trained model and vectoriser to the models/ directory
  7. Save evaluation results to outputs/evaluation/ for use by the Streamlit app
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

    # Drop any rows where text_combined is NaN — these would cause the
    # vectoriser to fail or produce meaningless zero vectors
    df = df.dropna(subset=["text_combined"])

    X = df["text_combined"]
    y = df["label"].astype(int)

    # Convert everything to string to prevent type errors during vectorisation
    # (e.g. if a cell was read as a float due to mixed column types)
    X = X.astype(str)

    # random_state=42 ensures the train/test split is identical every run,
    # which makes results reproducible — important for a fair evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # Logistic Regression was chosen over alternatives (e.g. SVM, Random Forest,
    # neural networks) because:
    #   - It is highly interpretable — coefficients map directly to word importance,
    #     which pairs well with TF-IDF sparse vectors
    #   - It trains quickly even on large sparse matrices, which is practical for
    #     a university project with limited compute
    #   - It produces well-calibrated probability estimates, making the confidence
    #     score meaningful to the end user
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    # predict_proba gives us the probability for each class; we take class 1
    # (phishing) for the ROC-AUC calculation
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    # --- Individual metric print statements ---
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
    # Rows = actual class, Columns = predicted class
    # [TN  FP]
    # [FN  TP]
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix (rows=Actual, cols=Predicted) ---")
    print(f"                Predicted Legitimate  Predicted Phishing")
    print(f"Actual Legitimate      {cm[0][0]:<20} {cm[0][1]}")
    print(f"Actual Phishing        {cm[1][0]:<20} {cm[1][1]}")

    # --- Full classification report ---
    report = classification_report(y_test, y_pred, digits=3)
    print("\n--- Full Classification Report ---")
    print(report)

    # --- Save evaluation outputs ---
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the full text report so it can be reviewed outside the terminal
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

    # Save individual metrics as JSON so the Streamlit app can load and display
    # them in the sidebar without needing to re-run training
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
