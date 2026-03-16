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
from explain import build_explainer, explain_with_lime, explain_with_coefficients

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
            # Display each metric as a percentage-style figure for readability,
            # followed by a plain English caption so non-technical users understand
            # what each number actually means
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            st.caption(
                "Out of every 100 emails, the model correctly identifies approximately "
                f"{metrics.get('accuracy', 0) * 100:.0f} as either phishing or legitimate."
            )

            st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            st.caption(
                "When the model flags an email as phishing, it is correct approximately "
                f"{metrics.get('precision', 0) * 100:.0f}% of the time."
            )

            st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            st.caption(
                "The model successfully catches approximately "
                f"{metrics.get('recall', 0) * 100:.0f} out of every 100 actual phishing emails."
            )

            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
            st.caption(
                "The overall balance between precision and recall. "
                "A score above 95% is considered excellent."
            )

            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            st.caption(
                "Measures how well the model separates phishing from legitimate emails. "
                "1.0 is a perfect score. Anything above 0.99 is exceptional."
            )
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

    weights = explain_with_lime(explainer, model, vec, combined_text, num_features=15)

    import pandas as pd

    # Plain English key so users know how to interpret the table without
    # needing any machine learning background
    st.info(
        "**How to read this table:** Each row shows a word or phrase from your "
        "email and how much it influenced the result. A **positive score** means "
        "that word pushed the classification towards **Phishing**. A **negative "
        "score** means it pushed towards **Legitimate**. The larger the number, "
        "the stronger the influence."
    )

    # Build a tidy dataframe sorted by how strongly each word influenced the
    # decision (positive = towards phishing, negative = towards legitimate)
    explanation_df = pd.DataFrame(weights, columns=["Word or Phrase", "Contribution Score"])
    explanation_df = explanation_df.reindex(
        explanation_df["Contribution Score"].abs().sort_values(ascending=False).index
    )
    explanation_df = explanation_df.reset_index(drop=True)
    # Round to 3 decimal places so the table is readable at a glance
    explanation_df["Contribution Score"] = explanation_df["Contribution Score"].round(3)
    st.dataframe(explanation_df, use_container_width=True)

    # --- Model Coefficient Analysis ---
    # A second complementary view: instead of LIME's local perturbation scores,
    # this shows the model's global learned weights multiplied by TF-IDF score,
    # giving a direct measure of each word's influence based on training data
    st.subheader("Model Coefficient Analysis")

    coef_weights = explain_with_coefficients(model, vec, combined_text, top_n=10)
    coef_df = pd.DataFrame(coef_weights, columns=["Word or Phrase", "Contribution Score"])
    coef_df = coef_df.reset_index(drop=True)
    # Round to 3 decimal places for consistency with the LIME table above
    coef_df["Contribution Score"] = coef_df["Contribution Score"].round(3)
    st.dataframe(coef_df, use_container_width=True)
    st.caption(
        "Positive scores indicate words associated with phishing across the entire "
        "training dataset. Negative scores indicate words associated with legitimate emails."
    )

    # Log only metadata — never the email content — for later evaluation
    ts = datetime.now().isoformat(timespec="seconds")
    log_line = f"{ts},{label},{confidence:.4f},{len(combined_text)}\n"
    if not LOG_FILE.exists():
        LOG_FILE.write_text("timestamp,prediction,confidence,text_length\n", encoding="utf-8")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(log_line)

    st.caption("Only metadata is logged locally (no email content stored).")
