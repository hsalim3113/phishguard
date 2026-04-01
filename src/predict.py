"""predict.py — model loading and inference for the phishing detector."""

import numpy as np
from joblib import load

from config import MAX_INPUT_CHARS, MODEL_LOGREG, VEC_PATH


def load_assets():
    """Load the trained Logistic Regression model and TF-IDF vectoriser from disk.

    Returns:
        tuple: (model, vec) where model is the fitted LogisticRegression and
            vec is the fitted TfidfVectorizer.

    Raises:
        FileNotFoundError: If model or vectoriser files are missing (run train.py).
    """
    model = load(MODEL_LOGREG)
    vec = load(VEC_PATH)
    return model, vec


def predict_email(model, vec, subject: str, body: str) -> tuple[str, float, str]:
    """Classify a single email as phishing or legitimate.

    The subject and body are concatenated, then truncated to ``MAX_INPUT_CHARS``
    before vectorisation.  This matches the cap enforced in the Streamlit app.

    Args:
        model: Fitted sklearn classifier with ``predict_proba``.
        vec: Fitted TF-IDF vectoriser.
        subject (str): Email subject line (may be empty).
        body (str): Email body text (may be empty).

    Returns:
        tuple[str, float, str]: A three-element tuple of
            (label, confidence, combined_text) where label is ``"phishing"``
            or ``"legitimate"``, confidence is the model probability in
            ``[0, 1]``, and combined_text is the (possibly truncated) text
            that was actually fed to the model.

    Raises:
        ValueError: If both subject and body are empty or whitespace only.
    """
    # Join subject and body into one string so the vectoriser sees them together,
    # which is the same format the model was trained on.
    combined = ((subject or "").strip() + "\n" + (body or "").strip()).strip()
    if not combined:
        raise ValueError("Both subject and body are empty.")

    # Cut off anything over the limit so the vectoriser doesn't get flooded
    if len(combined) > MAX_INPUT_CHARS:
        combined = combined[:MAX_INPUT_CHARS]

    X = vec.transform([combined])

    # predict_proba gives [P(legitimate), P(phishing)], take the highest as confidence
    proba = model.predict_proba(X)[0]
    # argmax picks whichever class (0=legitimate, 1=phishing) has the higher probability
    pred_idx = int(np.argmax(proba))
    confidence = float(np.max(proba))
    label = "phishing" if pred_idx == 1 else "legitimate"
    return label, confidence, combined
