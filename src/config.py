# config.py: centralised constants for the phishing detector project.
# All other modules import from here rather than hardcoding paths or values.

from pathlib import Path

# ---------------------------------------------------------------------------
# File paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_RAW = Path("data/raw/dataset.csv")
DATA_PROCESSED = Path("data/processed/processed.csv")

MODEL_LOGREG = Path("models/model_logreg.joblib")
MODEL_RF = Path("models/model_rf.joblib")
MODEL_NB = Path("models/model_nb.joblib")
VEC_PATH = Path("models/tfidf_vectorizer.joblib")

EVAL_DIR = Path("outputs/evaluation")
LOG_DIR = Path("outputs/logs")

# Derived evaluation file paths
CLASSIFICATION_REPORT_TXT = EVAL_DIR / "classification_report.txt"
CLASSIFICATION_REPORT_JSON = EVAL_DIR / "classification_report.json"
CONFUSION_MATRIX_PNG = EVAL_DIR / "confusion_matrix.png"
ROC_CURVE_PNG = EVAL_DIR / "roc_curve.png"
CROSS_VAL_TXT = EVAL_DIR / "cross_validation_results.txt"
MODEL_COMPARISON_CSV = EVAL_DIR / "model_comparison.csv"
MODEL_COMPARISON_PNG = EVAL_DIR / "model_comparison.png"
DATASET_STATS_TXT = EVAL_DIR / "dataset_stats.txt"
DATASET_STATS_JSON = EVAL_DIR / "dataset_stats.json"

# ---------------------------------------------------------------------------
# Model / vectoriser parameters
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Application parameters
# ---------------------------------------------------------------------------
MAX_INPUT_CHARS = 5000
LIME_NUM_FEATURES = 10
SHAP_NUM_FEATURES = 10
