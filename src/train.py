"""train.py — trains and evaluates Logistic Regression, Random Forest,
Naive Bayes, and XGBoost classifiers on TF-IDF features derived from
processed email data.

Four models are trained on the same feature set so their performance can be
directly compared. The comparison justifies choosing Logistic Regression as
the app's primary model.

Run from the project root:
    python src/train.py

Outputs
-------
models/
    model_logreg.joblib      — primary prediction model used by the app
    model_rf.joblib          — Random Forest (comparison only)
    model_nb.joblib          — Naive Bayes (comparison baseline)
    model_xgb.joblib         — XGBoost (used for SHAP TreeExplainer)
    tfidf_vectorizer.joblib  — must be loaded alongside any model
outputs/evaluation/
    classification_report.txt  — human-readable precision/recall/F1 for LogReg
    classification_report.json — same report in JSON for the Streamlit sidebar
    confusion_matrix.png       — 2×2 grid of correct/incorrect predictions
    roc_curve.png              — trade-off between sensitivity and specificity
    cross_validation_results.txt — 5-fold CV scores for LogReg
    model_comparison.csv       — one row per model with all test-set metrics
    model_comparison.png       — grouped bar chart of the above
"""

# json: saves the classification report as a file the Streamlit sidebar can parse.
import json

# time: measures how many seconds each model takes to train — shown in the comparison table.
import time

# warnings: suppresses verbose sklearn/XGBoost notices that don't affect results.
import warnings

# matplotlib draws the confusion matrix, ROC curve, and comparison bar chart.
# "Agg" backend renders charts to files rather than opening a window — required on servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# numpy: array operations for building bar chart positions and numeric calculations.
import numpy as np

# pandas: loads the processed CSV and builds the model comparison DataFrame.
import pandas as pd

# joblib.dump: saves a fitted model or vectoriser to a binary file for later reloading.
from joblib import dump

# The four classifier classes used for training.
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,   # per-class precision, recall, F1, and support
    confusion_matrix,        # 2×2 table of correct and incorrect classifications
    roc_auc_score,           # single number summarising the ROC curve (0.5–1.0)
    roc_curve,               # series of (FPR, TPR) points at each decision threshold
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

# XGBoost: gradient boosting — builds trees sequentially, each correcting the previous one.
# Included to enable the SHAP TreeExplainer explanation option in the app.
from xgboost import XGBClassifier

# All paths and parameters come from config.py so a change only needs to be made once.
from config import (
    CV_FOLDS,
    DATA_PROCESSED,
    EVAL_DIR,
    MODEL_LOGREG,
    MODEL_NB,
    MODEL_RF,
    MODEL_XGB,
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

# Suppress deprecation and convergence notices — informational only, do not affect results.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Evaluation helpers — each saves a single output file to outputs/evaluation/
# ---------------------------------------------------------------------------

def _save_classification_report(y_test, y_pred) -> None:
    """Compute and save the classification report for the Logistic Regression model.

    Reports four metrics per class: precision (how many predictions were correct),
    recall (how many actual positives were caught), F1 (harmonic mean of the two),
    and support (raw count). Weighted averages account for class imbalance.

    Args:
        y_test: True integer labels for the test set (0=legitimate, 1=phishing).
        y_pred: Predicted integer labels from the trained Logistic Regression.

    Returns:
        None. Writes to CLASSIFICATION_REPORT_TXT and CLASSIFICATION_REPORT_JSON.
    """
    # digits=4 gives enough decimal places to distinguish small differences between models.
    # target_names maps 0 → "Legitimate" and 1 → "Phishing" for readability.
    report_str = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Phishing"], digits=4
    )
    CLASSIFICATION_REPORT_TXT.write_text(report_str, encoding="utf-8")
    print("\nClassification Report (Logistic Regression):")
    print(report_str)

    # output_dict=True returns the same data as a nested dict so the Streamlit app
    # can read individual values like report["accuracy"] programmatically.
    report_dict = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Phishing"],
        output_dict=True
    )
    CLASSIFICATION_REPORT_JSON.write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )


def _save_confusion_matrix(y_test, y_pred) -> None:
    """Plot and save a confusion matrix as a PNG image.

    The 2×2 matrix shows true negatives (correct legitimate), false positives
    (legitimate wrongly flagged), false negatives (missed phishing), and true
    positives (correctly caught phishing). In a security context, false negatives
    are more dangerous than false positives — this chart shows how the model balances them.

    Args:
        y_test: True labels for the test split.
        y_pred: Predicted labels from the model.

    Returns:
        None. Saves the PNG to CONFUSION_MATRIX_PNG.
    """
    # confusion_matrix() returns a 2×2 array; rows = actual class, columns = predicted class.
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    # imshow renders the matrix as a colour grid — darker blue = higher count.
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)  # colour scale so users know what darkness means
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ylabel="True label",   # rows = what the email actually was
        xlabel="Predicted label",  # columns = what the model predicted
        title="Confusion Matrix — Logistic Regression",
    )

    # Overlay the count in each cell; white text on dark cells, black on light cells.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=120)  # dpi=120 gives a crisp image in the sidebar
    plt.close(fig)  # free memory — matplotlib holds figures until explicitly closed
    print(f"Saved: {CONFUSION_MATRIX_PNG}")


def _save_roc_curve(model, X_test_vec, y_test) -> None:
    """Compute and save the ROC curve and AUC score for the Logistic Regression model.

    The ROC curve plots how many phishing emails are caught (true positive rate) versus
    how many legitimate emails are wrongly flagged (false positive rate) at every
    possible confidence threshold. AUC summarises this as one number — 1.0 is perfect,
    0.5 is random guessing. AUC is preferred over accuracy because it is threshold-independent.

    Args:
        model: Trained sklearn classifier with predict_proba support.
        X_test_vec: Vectorised test features (sparse TF-IDF matrix).
        y_test: True labels for the test split.

    Returns:
        None. Saves the PNG to ROC_CURVE_PNG and AUC value to auc_score.txt.
    """
    # [:, 1] takes the phishing probability column — the ROC curve is built on the
    # model's confidence that each email is the positive (phishing) class.
    y_prob = model.predict_proba(X_test_vec)[:, 1]

    # roc_curve() returns (FPR, TPR) pairs at every threshold; _ discards the threshold values.
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # AUC equals the probability that the model ranks a random phishing email
    # higher than a random legitimate one.
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC score (Logistic Regression): {auc:.4f}")

    # The diagonal dashed line is the random-guessing baseline (AUC = 0.5).
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)  # diagonal reference line (random baseline)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Logistic Regression")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved: {ROC_CURVE_PNG}")

    # Also write the AUC number to a text file for quick checking without opening the image.
    (EVAL_DIR / "auc_score.txt").write_text(
        f"AUC (Logistic Regression): {auc:.4f}\n", encoding="utf-8"
    )


def _run_cross_validation(model_class, X_vec, y) -> None:
    """Run k-fold cross-validation on the full dataset and save results.

    Splits the data into CV_FOLDS equal parts, trains on all but one part, tests
    on the remaining part, and rotates until every email has been tested once.
    This gives a more stable accuracy estimate than a single train/test split.

    Args:
        model_class: Unfitted sklearn estimator class (a new instance is created
            here so its weights are independent of the model trained in main()).
        X_vec: Full vectorised feature matrix (all emails, not split).
        y: Full label Series corresponding to X_vec.

    Returns:
        None. Writes fold scores, mean, and std to CROSS_VAL_TXT.
    """
    print(f"\nRunning {CV_FOLDS}-fold cross-validation …")

    # A fresh instance is created so its weights are not influenced by main()'s training.
    # max_iter=2000 ensures convergence on all folds.
    cv_model = model_class(max_iter=2000, random_state=RANDOM_STATE)

    # n_jobs=-1 runs all folds in parallel across available CPU cores.
    scores = cross_val_score(cv_model, X_vec, y, cv=CV_FOLDS, scoring="accuracy",
                             n_jobs=-1)

    # Mean = expected performance; std = how consistent the model is across folds.
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
    """Fit a classifier on the training data and return it with its training duration.

    Training time is included in the comparison table to show the speed trade-off
    between models (Logistic Regression takes seconds; XGBoost takes minutes).

    Args:
        model: Unfitted sklearn estimator — fitted in-place by this function.
        X_train_vec: Sparse TF-IDF matrix for the training set.
        y_train: Integer class labels for the training set.

    Returns:
        tuple: (fitted model, training time in seconds as a float).
    """
    # Record wall-clock time before and after .fit() to compute elapsed seconds.
    t0 = time.time()

    # .fit() learns the model's weights from the training data in-place.
    model.fit(X_train_vec, y_train)

    return model, time.time() - t0


def _build_comparison_row(name, model, X_test_vec, y_test, train_time):
    """Evaluate a fitted model on the test set and return its metrics as a dict.

    All four models are evaluated identically so the comparison is fair — same
    test set, same feature representation. Weighted averages account for any
    class imbalance in the dataset.

    Args:
        name (str): Display name for the model (e.g. "Random Forest").
        model: Fitted sklearn estimator.
        X_test_vec: Sparse TF-IDF matrix for the test set.
        y_test: True labels for the test set (0 or 1).
        train_time (float): Elapsed training seconds from _train_and_time.

    Returns:
        dict with keys: model, accuracy, precision_weighted, recall_weighted,
            f1_weighted, train_time_s. All metric values are rounded to 4 decimal places.
    """
    # .predict() applies the learned decision boundary to return 0 or 1 per email.
    y_pred = model.predict(X_test_vec)

    # output_dict=True returns per-class and weighted-average metrics in one call.
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
    """Save the four-model comparison as a CSV table and a grouped bar chart PNG.

    Both formats are produced: the CSV for detailed numeric inspection and the PNG
    for display in the Streamlit sidebar where users can visually compare models.

    Args:
        rows (list[dict]): One dict per model, each from _build_comparison_row.

    Returns:
        None. Writes to MODEL_COMPARISON_CSV and MODEL_COMPARISON_PNG.
    """
    # index=False omits pandas row numbers so the CSV has only the metric columns.
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"\nModel comparison:\n{comp_df.to_string(index=False)}")
    print(f"Saved: {MODEL_COMPARISON_CSV}")

    # Four metric groups along the x-axis, each with one bar per model.
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    # np.arange gives base x positions [0, 1, 2, 3]; each model's bars are offset by i * width.
    x = np.arange(len(metrics))

    # width=0.2 means 4 bars × 0.2 = 0.8 per group, leaving a 0.2 gap between groups.
    width = 0.2

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Each model gets its own set of bars, offset so they sit side by side per metric group.
    for i, row in enumerate(rows):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"], color=colours[i])

    # x + width * 1.5 centres the tick label under the middle of each group of 4 bars.
    ax.set_xticks(x + width * 1.5)
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
    """Full training pipeline: load data, vectorise, train 4 models, evaluate, save.

    Logistic Regression gets the deepest evaluation (confusion matrix, ROC curve,
    cross-validation) because it is the model the Streamlit app uses for predictions.
    All four models use the same vectoriser so the comparison is fair.

    Returns:
        None. All model files, charts, and reports are written to disk.

    Raises:
        FileNotFoundError: If the processed CSV does not exist — run preprocess.py first.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOGREG.parent.mkdir(parents=True, exist_ok=True)

    # Load the cleaned dataset — two columns: email text and integer label.
    df = pd.read_csv(DATA_PROCESSED)

    # X = text strings (model input); y = integer labels (model target).
    X = df["text_combined"].astype(str)
    y = df["label"].astype(int)

    # ---------------------------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------------------------
    # stratify=y keeps the same phishing-to-legitimate ratio in both halves —
    # without it an unlucky split could skew the test metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---------------------------------------------------------------------------
    # TF-IDF vectorisation
    # ---------------------------------------------------------------------------
    # TF-IDF converts email text to numbers. Each word gets a score based on how
    # often it appears in THIS email relative to all emails — rare but frequent words
    # score highest. stop_words="english" removes common words like "the" and "and"
    # that appear in every email and carry no discriminative signal.
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
    )

    # Fit on training data ONLY — fitting on all data before splitting is data leakage
    # (the model would cheat by knowing test vocabulary, making results look falsely good).
    X_train_vec = vec.fit_transform(X_train)  # learn vocabulary AND transform training data

    # transform() applies the already-learned vocabulary to test data without adding new words.
    # This simulates real production: the model will always see new vocabulary it hasn't learned.
    X_test_vec = vec.transform(X_test)

    # Transform the full dataset for cross-validation — no leakage because
    # the vectoriser is already fitted and cross_val_score handles its own splits.
    X_full_vec = vec.transform(X)

    # ---------------------------------------------------------------------------
    # Model definitions
    # ---------------------------------------------------------------------------

    # LOGISTIC REGRESSION — the primary app model.
    # Learns one weight per word; that weight directly represents the word's contribution
    # to the phishing probability. Fast, interpretable, and works well on sparse TF-IDF.
    # max_iter=2000 — the default 100 is often too few for a 30,000-word vocabulary.
    logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    # RANDOM FOREST — 200 decision trees that each vote on the class.
    # Included for comparison; typically slower and harder to explain than LogReg on sparse data.
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    # MULTINOMIAL NAIVE BAYES — a simple probabilistic model that trains almost instantly.
    # Included as a baseline to show how much accuracy the complex models gain over a simple one.
    nb = MultinomialNB()

    # XGBOOST — builds trees sequentially, each correcting the previous tree's errors.
    # Included primarily to enable the SHAP TreeExplainer option in the app.
    # Does not call .toarray() because XGBoost handles sparse matrices natively.
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------

    print("Training Logistic Regression …")
    logreg, t_lr = _train_and_time(logreg, X_train_vec, y_train)

    print("Training Random Forest …")
    rf, t_rf = _train_and_time(rf, X_train_vec, y_train)

    print("Training Multinomial Naive Bayes …")
    nb, t_nb = _train_and_time(nb, X_train_vec, y_train)

    print("Training XGBoost …")
    xgb, t_xgb = _train_and_time(xgb, X_train_vec, y_train)

    # ---------------------------------------------------------------------------
    # Save models and vectoriser to disk
    # ---------------------------------------------------------------------------
    # The vectoriser MUST be saved with the models because their weight matrices
    # are indexed to its vocabulary — loading either without the other breaks predictions.
    dump(logreg, MODEL_LOGREG)
    dump(rf, MODEL_RF)
    dump(nb, MODEL_NB)
    dump(xgb, MODEL_XGB)
    dump(vec, VEC_PATH)
    print(f"\nSaved: {MODEL_LOGREG}, {MODEL_RF}, {MODEL_NB}, {MODEL_XGB}, {VEC_PATH}")

    # ---------------------------------------------------------------------------
    # Deep evaluation — Logistic Regression only (the app's production model)
    # ---------------------------------------------------------------------------

    # Get predicted labels for the test set.
    y_pred_lr = logreg.predict(X_test_vec)

    # Save precision/recall/F1 report in both text and JSON formats.
    _save_classification_report(y_test, y_pred_lr)

    # Save the 2×2 confusion matrix.
    _save_confusion_matrix(y_test, y_pred_lr)

    # Save the ROC curve showing the sensitivity/specificity trade-off.
    _save_roc_curve(logreg, X_test_vec, y_test)

    # 5-fold CV on the full dataset gives a more stable accuracy estimate than one split.
    _run_cross_validation(LogisticRegression, X_full_vec, y)

    # ---------------------------------------------------------------------------
    # Model comparison table and chart
    # ---------------------------------------------------------------------------
    # All four models are evaluated on the same test set for a fair comparison.
    rows = [
        _build_comparison_row("Logistic Regression", logreg, X_test_vec, y_test, t_lr),
        _build_comparison_row("Random Forest", rf, X_test_vec, y_test, t_rf),
        _build_comparison_row("Naive Bayes", nb, X_test_vec, y_test, t_nb),
        _build_comparison_row("XGBoost", xgb, X_test_vec, y_test, t_xgb),
    ]
    _save_model_comparison(rows)

    # Logistic Regression is the app's production model: highest accuracy (~98.5%),
    # one weight per word (fully interpretable), and predicts in microseconds.


# Only run main() when the script is executed directly, not when imported.
if __name__ == "__main__":
    main()
