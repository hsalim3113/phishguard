"""train.py — trains and evaluates Logistic Regression, Random Forest, and
Naive Bayes classifiers on TF-IDF features derived from processed email data.

Run from the project root:
    python src/train.py

Outputs
-------
models/
    model_logreg.joblib
    model_rf.joblib
    model_nb.joblib
    tfidf_vectorizer.joblib
outputs/evaluation/
    classification_report.txt
    classification_report.json
    confusion_matrix.png
    roc_curve.png
    cross_validation_results.txt
    model_comparison.csv
    model_comparison.png
"""

import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

from config import (
    CV_FOLDS,
    DATA_PROCESSED,
    EVAL_DIR,
    MODEL_LOGREG,
    MODEL_NB,
    MODEL_RF,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    VEC_PATH,
    CLASSIFICATION_REPORT_JSON,
    CLASSIFICATION_REPORT_TXT,
    CONFUSION_MATRIX_PNG,
    CROSS_VAL_TXT,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_PNG,
    ROC_CURVE_PNG,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Evaluation helpers, each one saves a single output file to outputs/evaluation/
# ---------------------------------------------------------------------------

def _save_classification_report(y_test, y_pred) -> None:
    """Compute and persist the classification report as .txt and .json.

    Args:
        y_test: True labels for the test split.
        y_pred: Predicted labels from the model.

    Returns:
        None
    """
    report_str = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Phishing"], digits=4
    )
    CLASSIFICATION_REPORT_TXT.write_text(report_str, encoding="utf-8")
    print("\nClassification Report (Logistic Regression):")
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Phishing"],
        output_dict=True
    )
    CLASSIFICATION_REPORT_JSON.write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )


def _save_confusion_matrix(y_test, y_pred) -> None:
    """Plot and save a confusion matrix as a PNG.

    Args:
        y_test: True labels for the test split.
        y_pred: Predicted labels from the model.

    Returns:
        None
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix — Logistic Regression",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved: {CONFUSION_MATRIX_PNG}")


def _save_roc_curve(model, X_test_vec, y_test) -> None:
    """Compute and save the ROC curve and AUC score.

    Args:
        model: Trained sklearn classifier with predict_proba support.
        X_test_vec: Vectorised test features.
        y_test: True labels for the test split.

    Returns:
        None
    """
    y_prob = model.predict_proba(X_test_vec)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC score (Logistic Regression): {auc:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Logistic Regression")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved: {ROC_CURVE_PNG}")

    (EVAL_DIR / "auc_score.txt").write_text(
        f"AUC (Logistic Regression): {auc:.4f}\n", encoding="utf-8"
    )


def _run_cross_validation(model_class, X_vec, y) -> None:
    """Run 5-fold stratified cross-validation and save results.

    Args:
        model_class: Unfitted sklearn estimator class to evaluate.
        X_vec: Full vectorised feature matrix.
        y: Full label series.

    Returns:
        None
    """
    print(f"\nRunning {CV_FOLDS}-fold cross-validation …")
    cv_model = model_class(max_iter=2000, random_state=RANDOM_STATE)
    scores = cross_val_score(cv_model, X_vec, y, cv=CV_FOLDS, scoring="accuracy",
                             n_jobs=-1)
    mean_acc = scores.mean()
    std_acc = scores.std()
    result = (
        f"{CV_FOLDS}-Fold Cross-Validation Results (Logistic Regression)\n"
        f"{'=' * 50}\n"
        f"Fold scores : {', '.join(f'{s:.4f}' for s in scores)}\n"
        f"Mean accuracy: {mean_acc:.4f}\n"
        f"Std deviation: {std_acc:.4f}\n"
    )
    print(result)
    CROSS_VAL_TXT.write_text(result, encoding="utf-8")
    print(f"Saved: {CROSS_VAL_TXT}")


def _train_and_time(model, X_train_vec, y_train):
    """Fit a model and return it along with elapsed training time.

    Args:
        model: Unfitted sklearn estimator.
        X_train_vec: Vectorised training features.
        y_train: Training labels.

    Returns:
        tuple: (fitted model, training time in seconds as float)
    """
    t0 = time.time()
    model.fit(X_train_vec, y_train)
    return model, time.time() - t0


def _build_comparison_row(name, model, X_test_vec, y_test, train_time):
    """Compute evaluation metrics for one model and return as a dict.

    Args:
        name (str): Display name for the model.
        model: Fitted sklearn estimator.
        X_test_vec: Vectorised test features.
        y_test: True test labels.
        train_time (float): Elapsed training time in seconds.

    Returns:
        dict: Keys — model, accuracy, precision, recall, f1, train_time_s.
    """
    y_pred = model.predict(X_test_vec)
    rep = classification_report(y_test, y_pred, output_dict=True)
    return {
        "model": name,
        "accuracy": round(rep["accuracy"], 4),
        "precision_weighted": round(rep["weighted avg"]["precision"], 4),
        "recall_weighted": round(rep["weighted avg"]["recall"], 4),
        "f1_weighted": round(rep["weighted avg"]["f1-score"], 4),
        "train_time_s": round(train_time, 3),
    }


def _save_model_comparison(rows) -> None:
    """Save model comparison table as CSV and a grouped bar chart PNG.

    Args:
        rows (list[dict]): One dict per model from _build_comparison_row.

    Returns:
        None
    """
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"\nModel comparison:\n{comp_df.to_string(index=False)}")
    print(f"Saved: {MODEL_COMPARISON_CSV}")

    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    x = np.arange(len(metrics))
    width = 0.25
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in enumerate(rows):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"], color=colours[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
                         "F1\n(weighted)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_COMPARISON_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved: {MODEL_COMPARISON_PNG}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Full training pipeline: load data, vectorise, train 3 models, evaluate.

    Returns:
        None

    Raises:
        FileNotFoundError: If the processed CSV does not exist (run preprocess.py first).
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOGREG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PROCESSED)
    X = df["text_combined"].astype(str)
    y = df["label"].astype(int)

    # 80/20 split. stratify=y keeps the same class ratio in both halves,
    # and random_state=42 means the split is the same every time you run it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # TF-IDF turns the text into a numeric feature vector. ngram_range=(1,2)
    # means it looks at single words and two-word phrases like "click here",
    # which tend to be more useful than individual words on their own.
    # The vectoriser is only fitted on training data so the test set stays unseen.
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
    )
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    X_full_vec = vec.transform(X)

    # All three models use the same split so the comparison is fair
    logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    nb = MultinomialNB()

    print("Training Logistic Regression …")
    logreg, t_lr = _train_and_time(logreg, X_train_vec, y_train)

    print("Training Random Forest …")
    rf, t_rf = _train_and_time(rf, X_train_vec, y_train)

    print("Training Multinomial Naive Bayes …")
    nb, t_nb = _train_and_time(nb, X_train_vec, y_train)

    dump(logreg, MODEL_LOGREG)
    dump(rf, MODEL_RF)
    dump(nb, MODEL_NB)
    dump(vec, VEC_PATH)
    print(f"\nSaved: {MODEL_LOGREG}, {MODEL_RF}, {MODEL_NB}, {VEC_PATH}")

    # Full evaluation on Logistic Regression since that's what the app uses
    y_pred_lr = logreg.predict(X_test_vec)
    _save_classification_report(y_test, y_pred_lr)
    _save_confusion_matrix(y_test, y_pred_lr)
    _save_roc_curve(logreg, X_test_vec, y_test)
    _run_cross_validation(LogisticRegression, X_full_vec, y)

    rows = [
        _build_comparison_row("Logistic Regression", logreg, X_test_vec, y_test, t_lr),
        _build_comparison_row("Random Forest", rf, X_test_vec, y_test, t_rf),
        _build_comparison_row("Naive Bayes", nb, X_test_vec, y_test, t_nb),
    ]
    _save_model_comparison(rows)

    # Logistic Regression is the model the app actually uses. The main reasons:
    # LIME and SHAP both work well with it because it gives a weight per feature.
    # It also trains in seconds compared to Random Forest, and predict_proba
    # gives a probability that's actually meaningful to show the user.
    # Random Forest is saved for comparison but isn't loaded by the app.


if __name__ == "__main__":
    main()
