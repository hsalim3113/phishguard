from pathlib import Path
from joblib import load
import numpy as np

MODEL_PATH = Path("models/model_logreg.joblib")
VEC_PATH = Path("models/tfidf_vectorizer.joblib")


def load_assets():
    """
    Load the trained model and TF-IDF vectoriser from disk.

    Both files are saved by train.py using joblib serialisation.
    This function is called once at app startup (cached by Streamlit) so
    that every prediction request reuses the same in-memory objects rather
    than reading from disk each time.

    Returns
    -------
    tuple
        (model, vec) where:
          - model : sklearn LogisticRegression — the trained classifier
          - vec   : sklearn TfidfVectorizer — the fitted vectoriser
    """
    model = load(MODEL_PATH)
    vec = load(VEC_PATH)
    return model, vec


def predict_email(model, vec, subject: str, body: str):
    """
    Predict whether an email is phishing or legitimate.

    Parameters
    ----------
    model : sklearn LogisticRegression
        The trained classifier returned by load_assets().
    vec : sklearn TfidfVectorizer
        The fitted vectoriser returned by load_assets().
    subject : str
        The email subject line (may be empty string but not both subject
        and body empty at the same time).
    body : str
        The email body text (may be empty string but not both subject
        and body empty at the same time).

    Returns
    -------
    tuple
        (label, confidence, combined_text) where:
          - label          : str   — 'phishing' or 'legitimate'
          - confidence     : float — probability of the predicted class (0–1)
          - combined_text  : str   — the joined text that was passed to the model

    Raises
    ------
    ValueError
        If both subject and body are empty strings after stripping whitespace.
    """
    # Validate input early so the rest of the function can assume non-empty text
    if not (subject or "").strip() and not (body or "").strip():
        raise ValueError("Email subject and body cannot both be empty")

    # Subject and body are joined with a newline so the vectoriser sees them as
    # a single document, which is consistent with how the training data was
    # structured in preprocess.py (text_combined = subject + "\n" + body)
    text = (subject or "").strip() + "\n" + (body or "").strip()

    X = vec.transform([text])
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    conf = float(np.max(proba))
    label = "phishing" if pred_idx == 1 else "legitimate"
    return label, conf, text
