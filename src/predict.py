from pathlib import Path
from joblib import load
import numpy as np

MODEL_PATH = Path("models/model_logreg.joblib")
VEC_PATH = Path("models/tfidf_vectorizer.joblib")


def load_assets():
    """
    Load the trained model and TF-IDF vectoriser from disk.

    Both files are saved by train.py using joblib. In the Streamlit app this
    function is wrapped in @st.cache_resource, so it only reads from disk once
    per session rather than on every button click.

    Returns
    -------
    tuple
        (model, vec) where:
          - model : sklearn LogisticRegression — the trained classifier
          - vec   : sklearn TfidfVectorizer    — the fitted vectoriser
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
        The email subject line. Can be empty, but not both subject and body.
    body : str
        The email body text. Can be empty, but not both subject and body.

    Returns
    -------
    tuple
        (label, confidence, combined_text) where:
          - label         : str   — 'phishing' or 'legitimate'
          - confidence    : float — probability of the predicted class (0–1)
          - combined_text : str   — the text that was actually passed to the model

    Raises
    ------
    ValueError
        If both subject and body are empty strings after stripping whitespace.
    """
    # Catch empty input before it reaches the model — an empty string would
    # produce a valid but completely meaningless TF-IDF vector full of zeros
    if not (subject or "").strip() and not (body or "").strip():
        raise ValueError("Email subject and body cannot both be empty")

    # Join subject and body the same way the training data was constructed in
    # preprocess.py — if the format here differs from training, the model is
    # effectively seeing a slightly different type of input than it was trained on
    text = (subject or "").strip() + "\n" + (body or "").strip()

    # Vectorise using the same fitted vectoriser as training
    X = vec.transform([text])

    # predict_proba returns [P(legitimate), P(phishing)] — argmax gives the
    # index of the winning class, and max gives the confidence score
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    conf = float(np.max(proba))
    label = "phishing" if pred_idx == 1 else "legitimate"
    return label, conf, text
