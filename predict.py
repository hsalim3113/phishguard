from pathlib import Path
from joblib import load
import numpy as np

MODEL_PATH = Path("models/model_logreg.joblib")
VEC_PATH = Path("models/tfidf_vectorizer.joblib")

def load_assets():
    model = load(MODEL_PATH)
    vec = load(VEC_PATH)
    return model, vec

def predict_email(model, vec, subject: str, body: str):
    text = (subject or "").strip() + "\n" + (body or "").strip()
    X = vec.transform([text])
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    conf = float(np.max(proba))
    label = "phishing" if pred_idx == 1 else "legitimate"
    return label, conf, text
