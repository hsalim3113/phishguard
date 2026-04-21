# config.py — centralised constants for the PhishGuard phishing detector project.
# All other modules import from here so a path or value only needs changing once.

# pathlib.Path builds file paths that work on Windows, macOS, and Linux.
from pathlib import Path

# ---------------------------------------------------------------------------
# File paths (relative to the project root)
# ---------------------------------------------------------------------------

# Original, uncleaned dataset — place the downloaded CSV here before running preprocess.py.
DATA_RAW = Path("data/raw/dataset.csv")

# Cleaned CSV written by preprocess.py — this is what train.py reads.
DATA_PROCESSED = Path("data/processed/processed.csv")

# ---------------------------------------------------------------------------
# Saved model files
# ---------------------------------------------------------------------------

# Primary prediction model — the Streamlit app loads this for every email prediction.
MODEL_LOGREG = Path("models/model_logreg.joblib")

# Saved for model comparison only — not used by the live app.
MODEL_RF = Path("models/model_rf.joblib")

# Saved for model comparison only — a simple baseline, not used by the live app.
MODEL_NB = Path("models/model_nb.joblib")

# XGBoost model — loaded when the user selects the SHAP (XGBoost) explanation method.
MODEL_XGB = Path("models/model_xgb.joblib")

# ---------------------------------------------------------------------------
# TF-IDF vectoriser path
# ---------------------------------------------------------------------------

# The vectoriser fitted during training. The models' weights are tied to its
# vocabulary, so this exact object must be saved and reloaded with every model.
VEC_PATH = Path("models/tfidf_vectorizer.joblib")

# ---------------------------------------------------------------------------
# Evaluation output paths — all derived from EVAL_DIR
# ---------------------------------------------------------------------------

# Root folder for all evaluation charts and reports.
EVAL_DIR = Path("outputs/evaluation")

# Folder for the local prediction log CSV written by the Streamlit app.
LOG_DIR = Path("outputs/logs")

# Human-readable text version of the classification report.
CLASSIFICATION_REPORT_TXT = EVAL_DIR / "classification_report.txt"

# JSON version of the same report — the Streamlit sidebar reads this to display metrics.
CLASSIFICATION_REPORT_JSON = EVAL_DIR / "classification_report.json"

# 2×2 grid showing how many emails were correctly and incorrectly classified.
CONFUSION_MATRIX_PNG = EVAL_DIR / "confusion_matrix.png"

# ROC curve — shows the trade-off between catching phishing and raising false alarms.
# AUC summarises this as one number (1.0 = perfect, 0.5 = random guessing).
ROC_CURVE_PNG = EVAL_DIR / "roc_curve.png"

# Text file storing fold-by-fold cross-validation accuracy scores.
CROSS_VAL_TXT = EVAL_DIR / "cross_validation_results.txt"

# CSV with one row per model — used to justify choosing Logistic Regression.
MODEL_COMPARISON_CSV = EVAL_DIR / "model_comparison.csv"

# Grouped bar chart of the model comparison — displayed in the Streamlit sidebar.
MODEL_COMPARISON_PNG = EVAL_DIR / "model_comparison.png"

# Human-readable dataset statistics file.
DATASET_STATS_TXT = EVAL_DIR / "dataset_stats.txt"

# JSON dataset statistics — the Streamlit app reads this for the "About" panel.
DATASET_STATS_JSON = EVAL_DIR / "dataset_stats.json"

# ---------------------------------------------------------------------------
# TF-IDF vectoriser hyperparameters
# ---------------------------------------------------------------------------

# Keep the 30,000 most informative word/phrase tokens — large enough to cover
# phishing vocabulary without using too much memory.
TFIDF_MAX_FEATURES = 30000

# Capture single words AND two-word phrases (bigrams). Bigrams like "click here"
# are stronger phishing signals than either word alone.
TFIDF_NGRAM_RANGE = (1, 2)

# ---------------------------------------------------------------------------
# Train / test split parameter
# ---------------------------------------------------------------------------

# Hold back 20% of emails as the test set — the model never sees these during training.
TEST_SIZE = 0.2

# Fixed random seed so every run produces the same split and weights, making results reproducible.
RANDOM_STATE = 42

# Split the data into 5 equal parts for cross-validation — each part acts as the
# test set once, giving 5 independent accuracy scores that are averaged.
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Application runtime parameters
# ---------------------------------------------------------------------------

# Emails longer than this are truncated — covers virtually all real emails
# while preventing very large pastes from slowing the vectoriser.
MAX_INPUT_CHARS = 5000

# Number of top-weighted words shown in the LIME explanation bar chart.
LIME_NUM_FEATURES = 10

# Number of top-weighted words shown in the SHAP explanation bar chart.
# Kept the same as LIME_NUM_FEATURES so both charts are directly comparable.
SHAP_NUM_FEATURES = 10
