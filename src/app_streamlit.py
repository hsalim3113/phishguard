# app_streamlit.py — PhishGuard Streamlit web application.
#
# This file defines the full user-facing interface for the PhishGuard phishing
# detector. It loads the pre-trained Logistic Regression model and TF-IDF
# vectoriser, accepts a user-supplied email subject and body, runs the
# prediction pipeline, and displays the result along with a LIME-based
# word-level explanation. An optional sidebar shows model metadata and
# evaluation metrics loaded from outputs/evaluation/metrics.json (produced
# by train.py). No email content is ever stored or transmitted externally.

import json
import streamlit as st
from datetime import datetime
from pathlib import Path

from predict import load_assets, predict_email
from explain import build_explainer, explain_with_lime

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions_log.csv"

METRICS_PATH = Path("outputs/evaluation/metrics.json")

st.set_page_config(page_title="PhishGuard: AI-Powered Phishing Email Detector", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.title("PhishGuard")
    st.markdown(
        """
        **About this tool**

        PhishGuard uses a Logistic Regression classifier trained on TF-IDF
        (Term Frequency–Inverse Document Frequency) features extracted from
        email text. The model was trained on a combined dataset drawn from
        the **Enron email corpus**, a **Kaggle phishing email dataset**, and
        **PhishTank** URL-based phishing samples. Logistic Regression was
        chosen for its interpretability and efficiency on sparse TF-IDF vectors.

        **Privacy:** No email data is stored or transmitted. Only anonymous
        metadata (timestamp, prediction label, confidence, text length) is
        logged locally for evaluation purposes.
        """
    )

    # Try to load metrics saved by train.py; show them if available
    st.markdown("---")
    st.subheader("Model Performance")
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            # Display each metric as a percentage-style figure for readability
            st.metric("Accuracy",  f"{metrics.get('accuracy',  0):.2%}")
            st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            st.metric("Recall",    f"{metrics.get('recall',    0):.2%}")
            st.metric("F1-Score",  f"{metrics.get('f1_score',  0):.2%}")
            st.metric("ROC-AUC",   f"{metrics.get('roc_auc',   0):.4f}")
        except (json.JSONDecodeError, KeyError):
            st.caption("Metrics file could not be parsed.")
    else:
        st.caption("No metrics found. Run train.py to generate them.")

# --- Main interface ---
st.title("PhishGuard: AI-Powered Phishing Email Detector")

# Brief description so users know what to do without reading any documentation
st.write(
    "Paste an email subject and body below to check whether it looks like a "
    "phishing attempt or a legitimate message. "
    "The tool will highlight the words that most influenced the decision."
)

@st.cache_resource
def get_assets():
    model, vec = load_assets()
    explainer = build_explainer()
    return model, vec, explainer

model, vec, explainer = get_assets()

subject = st.text_input("Email subject", "")
body = st.text_area("Email body", height=220)

learning_mode = st.checkbox("Learning mode (guess first)")
user_guess = None
if learning_mode:
    user_guess = st.radio("Your guess", ["phishing", "legitimate"], horizontal=True)

if st.button("Analyse email"):
    if not subject.strip() and not body.strip():
        st.error("Please enter a subject or a body.")
        st.stop()

    label, confidence, combined_text = predict_email(model, vec, subject, body)

    st.subheader("Result")

    # Use colour-coded banners so the verdict is immediately obvious
    if label == "phishing":
        st.error(f"Phishing Email Detected")
    else:
        st.success(f"Legitimate Email")

    # Show confidence as a percentage with a visual bar so users can gauge
    # how certain the model is — a 51% confidence is very different from 99%
    confidence_pct = confidence * 100
    st.write(f"**Confidence:** {confidence_pct:.2f}%")
    st.progress(confidence)  # st.progress expects a value between 0.0 and 1.0

    if learning_mode and user_guess is not None:
        st.subheader("Learning mode feedback")
        st.write(f"Your guess: {user_guess}")
        st.write("Match" if user_guess == label else "Mismatch")

    st.subheader("Explanation (top contributing words)")

    weights = explain_with_lime(explainer, model, vec, combined_text, num_features=10)

    # Build a tidy dataframe sorted by how strongly each word influenced the
    # decision (positive = towards phishing, negative = towards legitimate)
    import pandas as pd
    explanation_df = pd.DataFrame(weights, columns=["Word or Phrase", "Contribution Score"])
    explanation_df = explanation_df.reindex(
        explanation_df["Contribution Score"].abs().sort_values(ascending=False).index
    )
    explanation_df = explanation_df.reset_index(drop=True)
    st.dataframe(explanation_df, use_container_width=True)

    # Log only metadata — never the email content — for later evaluation
    ts = datetime.now().isoformat(timespec="seconds")
    log_line = f"{ts},{label},{confidence:.4f},{len(combined_text)}\n"
    if not LOG_FILE.exists():
        LOG_FILE.write_text("timestamp,prediction,confidence,text_length\n", encoding="utf-8")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(log_line)

    st.caption("Only metadata is logged locally (no email content stored).")
