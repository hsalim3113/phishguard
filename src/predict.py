"""predict.py — loads the trained model and vectoriser and classifies a single email."""

# numpy: np.argmax() and np.max() operate on predict_proba()'s output array
# to find the winning class index and its probability.
import numpy as np

# joblib.load: deserialises a .joblib file back into the Python object saved by train.py.
from joblib import load

# Centralised file paths and the input character limit.
from config import MAX_INPUT_CHARS, MODEL_LOGREG, VEC_PATH


def load_assets():
    """Load the trained Logistic Regression model and TF-IDF vectoriser from disk.

    Returns:
        tuple: (model, vec) — the fitted LogisticRegression and TfidfVectorizer objects.

    Raises:
        FileNotFoundError: If the joblib files don't exist (train.py not yet run).
    """
    model = load(MODEL_LOGREG)

    # MUST be the exact same vectoriser saved during training — same vocabulary and IDF values.
    vec = load(VEC_PATH)
    return model, vec


def predict_email(model, vec, subject: str, body: str) -> tuple[str, float, str]:
    """Classify a single email as phishing or legitimate using Logistic Regression.

    Args:
        model: Fitted LogisticRegression loaded by load_assets().
        vec: Fitted TfidfVectorizer loaded by load_assets().
        subject (str): Email subject line (may be empty).
        body (str): Email body text (may be empty).

    Returns:
        tuple[str, float, str]: (label, confidence, combined_text) where label is
            "phishing" or "legitimate", confidence is the winning class probability,
            and combined_text is the exact text fed to the model (post-truncation).

    Raises:
        ValueError: If both subject and body are empty.
    """
    # Combine subject and body the same way the training data was prepared.
    combined = ((subject or "").strip() + "\n" + (body or "").strip()).strip()

    if not combined:
        raise ValueError("Both subject and body are empty.")

    # Truncate to MAX_INPUT_CHARS — most phishing signal is in the first few paragraphs
    # and very long inputs slow down vectorisation without adding accuracy.
    if len(combined) > MAX_INPUT_CHARS:
        combined = combined[:MAX_INPUT_CHARS]

    # vec.transform([combined]) takes a list of one string — passing a bare string
    # would iterate over characters instead of treating it as one document.
    X = vec.transform([combined])

    # predict_proba() returns [[P(legitimate), P(phishing)]]; [0] takes the only row.
    proba = model.predict_proba(X)[0]

    # argmax returns the index of the class with the higher probability.
    pred_idx = int(np.argmax(proba))

    # float() converts numpy float64 to plain Python float for JSON/Firestore compatibility.
    confidence = float(np.max(proba))

    label = "phishing" if pred_idx == 1 else "legitimate"

    # Return combined (not the original subject/body) so the explainer uses identical input.
    return label, confidence, combined
