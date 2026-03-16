# app_streamlit.py — PhishGuard Streamlit web application.
#
# This is the main UI file. It loads the trained model and vectoriser,
# takes an email subject and body from the user, runs the prediction,
# and shows the result alongside two different word-level explanations
# (LIME and coefficient-based). The sidebar shows model metadata and the
# evaluation metrics produced by train.py.
#
# Learning Mode turns the tool into an interactive exercise — the result
# is hidden until the user submits their own guess, then feedback, scores,
# and explanations are revealed together.
#
# No email content is ever stored or transmitted. Only anonymous metadata
# (timestamp, label, confidence, text length) is logged locally.

import json
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

from predict import load_assets, predict_email
from explain import build_explainer, explain_with_lime, explain_with_coefficients

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions_log.csv"

METRICS_PATH = Path("outputs/evaluation/metrics.json")

# --- Session state ---
# Streamlit reruns the entire script on every interaction, so any state we
# want to keep between button clicks has to live in st.session_state.
# These variables track the Learning Mode score and control whether the
# result has been revealed yet. The "not in" check means they only get
# initialised on the very first run, not overwritten on subsequent reruns.
if "lm_score" not in st.session_state:
    st.session_state.lm_score = 0
if "lm_attempts" not in st.session_state:
    st.session_state.lm_attempts = 0
if "lm_result_revealed" not in st.session_state:
    st.session_state.lm_result_revealed = False
if "lm_last_correct" not in st.session_state:
    st.session_state.lm_last_correct = None

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

    # Load the metrics file written by train.py — it won't exist if training
    # hasn't been run yet, so we check before trying to open it
    st.markdown("---")
    st.subheader("Model Performance")
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            # Show each metric with a plain English caption underneath —
            # the raw numbers don't mean much to a non-technical user without context
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

# Short description so users know what to do straight away —
# helpful for demos where there's no time to explain the context
st.write(
    "Paste an email subject and body below to check whether it looks like a "
    "phishing attempt or a legitimate message. "
    "The tool will highlight the words that most influenced the decision."
)

# @st.cache_resource means this only runs once per session — loading the
# model on every button click would be noticeably slow
@st.cache_resource
def get_assets():
    model, vec = load_assets()
    explainer = build_explainer()
    return model, vec, explainer

model, vec, explainer = get_assets()

# --- Example email loader ---
# Handy for demos and for testing Learning Mode without having to paste in
# a real email every time
EXAMPLE_SUBJECT = "URGENT: Your account has been suspended"
EXAMPLE_BODY = (
    "Dear Valued Customer, We have detected suspicious login activity on your account. "
    "To prevent unauthorised access, your account has been temporarily suspended pending "
    "verification. You must verify your identity immediately by clicking the secure link "
    "below. Failure to complete verification within 24 hours will result in permanent "
    "account closure and loss of all associated data. "
    "Click here to verify your account now: http://secure-account-verify-login.com/restore. "
    "Regards, Account Security Team."
)

if st.button("Load a Phishing Email Example"):
    # Setting these before st.rerun() means the values are already in session
    # state when the text_input and text_area widgets render on the next run
    st.session_state.subject_input = EXAMPLE_SUBJECT
    st.session_state.body_input = EXAMPLE_BODY
    # Clear any previous result so loading a new example doesn't show a stale verdict
    st.session_state.lm_result_revealed = False
    st.session_state.lm_last_correct = None
    st.rerun()

# The key= argument ties each widget's value to session state, which lets us
# pre-fill or clear the fields programmatically from button handlers elsewhere
subject = st.text_input("Email subject", "", key="subject_input")
body = st.text_area("Email body", "", height=220, key="body_input")

learning_mode = st.checkbox("Learning mode (guess first)")

# =====================================================================
# LEARNING MODE
# =====================================================================
if learning_mode:
    user_guess = st.radio(
        "Your guess — is this email phishing or legitimate?",
        ["phishing", "legitimate"],
        horizontal=True,
        key="user_guess_radio",
    )

    if st.button("Submit My Guess"):
        if not subject.strip() and not body.strip():
            st.error("Please enter a subject or body before submitting.")
            st.stop()

        # Run the prediction and both explainers, then store everything in
        # session state — without this, the results would be lost when Streamlit
        # reruns the script after the button click
        label, confidence, combined_text = predict_email(model, vec, subject, body)
        weights = explain_with_lime(explainer, model, vec, combined_text, num_features=15)
        coef_weights = explain_with_coefficients(model, vec, combined_text, top_n=10)

        st.session_state.lm_label = label
        st.session_state.lm_confidence = confidence
        st.session_state.lm_combined_text = combined_text
        st.session_state.lm_weights = weights
        st.session_state.lm_coef_weights = coef_weights

        # Compare the user's guess to the model's label and update the score
        correct = (user_guess == label)
        if correct:
            st.session_state.lm_score += 1
        st.session_state.lm_attempts += 1
        st.session_state.lm_last_correct = correct
        st.session_state.lm_result_revealed = True

        # Log here rather than after the reveal block — otherwise the entry
        # could be missed if the user closes the page before the result shows
        ts = datetime.now().isoformat(timespec="seconds")
        log_line = f"{ts},{label},{confidence:.4f},{len(combined_text)}\n"
        if not LOG_FILE.exists():
            LOG_FILE.write_text("timestamp,prediction,confidence,text_length\n", encoding="utf-8")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(log_line)

    # --- Reveal section ---
    # Everything below only appears once the user has hit Submit My Guess.
    # We pull the stored results back out of session state to display them.
    if st.session_state.lm_result_revealed:
        label        = st.session_state.lm_label
        confidence   = st.session_state.lm_confidence
        weights      = st.session_state.lm_weights
        coef_weights = st.session_state.lm_coef_weights

        # Running score — shows progress across the whole session
        st.metric(
            "Session Score",
            f"{st.session_state.lm_score} / {st.session_state.lm_attempts}",
        )

        # Immediate feedback on whether the guess was right
        if st.session_state.lm_last_correct:
            st.success("Correct! Well done.")
        else:
            st.error("Incorrect. See the explanation below to understand why.")

        # Model verdict — green for legitimate, red for phishing
        st.subheader("Model Result")
        if label == "phishing":
            st.error("Phishing Email Detected")
        else:
            st.success("Legitimate Email")

        # Confidence as a percentage plus a visual bar — a 51% confident
        # phishing label is very different from a 99% one
        confidence_pct = confidence * 100
        st.write(f"**Confidence:** {confidence_pct:.2f}%")
        st.progress(confidence)  # expects a float between 0.0 and 1.0

        # LIME explanation table — local to this specific email
        st.subheader("Explanation (top contributing words)")
        st.info(
            "**How to read this table:** Each row shows a word or phrase from your "
            "email and how much it influenced the result. A **positive score** means "
            "that word pushed the classification towards **Phishing**. A **negative "
            "score** means it pushed towards **Legitimate**. The larger the number, "
            "the stronger the influence."
        )
        explanation_df = pd.DataFrame(weights, columns=["Word or Phrase", "Contribution Score"])
        explanation_df = explanation_df.reindex(
            explanation_df["Contribution Score"].abs().sort_values(ascending=False).index
        )
        explanation_df = explanation_df.reset_index(drop=True)
        # Three decimal places is precise enough without making the table hard to read
        explanation_df["Contribution Score"] = explanation_df["Contribution Score"].round(3)
        st.dataframe(explanation_df, use_container_width=True)

        # Coefficient analysis — global view across the whole training dataset
        st.subheader("Model Coefficient Analysis")
        coef_df = pd.DataFrame(coef_weights, columns=["Word or Phrase", "Contribution Score"])
        coef_df = coef_df.reset_index(drop=True)
        coef_df["Contribution Score"] = coef_df["Contribution Score"].round(3)
        st.info(
            "**How to read this table:** Each row shows a word from the training data "
            "and its learned model weight. A **positive score** means that word is a "
            "strong indicator of **Phishing** emails in general. A **negative score** "
            "means that word is strongly associated with **Legitimate** emails. The "
            "larger the absolute value, the more influential that word is across the "
            "entire dataset — not just this specific email."
        )
        st.dataframe(coef_df, use_container_width=True)
        st.caption(
            "Positive scores indicate words associated with phishing across the entire "
            "training dataset. Negative scores indicate words associated with legitimate emails."
        )

        # Takeaway box — gives the user something concrete to remember
        st.info(
            "**What did you learn?**\n\n"
            "Phishing emails often contain urgent language, requests to verify personal "
            "details, suspicious links, and threats of account closure. Words like "
            "'urgent', 'verify', 'suspended', 'click', and 'immediately' are strong "
            "phishing indicators. Legitimate emails tend to use neutral, conversational "
            "language without pressure tactics."
        )

        # Reset button — clears the fields and hides the result so the user
        # can try another email without having to refresh the whole page
        if st.button("Try Another Email"):
            st.session_state.lm_result_revealed = False
            st.session_state.lm_last_correct = None
            st.session_state.subject_input = ""
            st.session_state.body_input = ""
            st.rerun()

        st.caption("Only metadata is logged locally (no email content stored).")

# =====================================================================
# NORMAL MODE
# =====================================================================
# Standard analysis flow — no score tracking, result shown immediately.
else:
    if st.button("Analyse email"):
        if not subject.strip() and not body.strip():
            st.error("Please enter a subject or a body.")
            st.stop()

        label, confidence, combined_text = predict_email(model, vec, subject, body)

        st.subheader("Result")

        # Green for legitimate, red for phishing — verdict is obvious at a glance
        if label == "phishing":
            st.error("Phishing Email Detected")
        else:
            st.success("Legitimate Email")

        # The confidence score matters — 99% confident is very different from 51%,
        # so we show both the percentage and a visual progress bar
        confidence_pct = confidence * 100
        st.write(f"**Confidence:** {confidence_pct:.2f}%")
        st.progress(confidence)  # st.progress expects a value between 0.0 and 1.0

        st.subheader("Explanation (top contributing words)")

        weights = explain_with_lime(explainer, model, vec, combined_text, num_features=15)

        # The scores only make sense if you know what they represent, so we
        # show a brief key before the table
        st.info(
            "**How to read this table:** Each row shows a word or phrase from your "
            "email and how much it influenced the result. A **positive score** means "
            "that word pushed the classification towards **Phishing**. A **negative "
            "score** means it pushed towards **Legitimate**. The larger the number, "
            "the stronger the influence."
        )

        # Convert the LIME output to a sorted dataframe so the most influential
        # words appear at the top
        explanation_df = pd.DataFrame(weights, columns=["Word or Phrase", "Contribution Score"])
        explanation_df = explanation_df.reindex(
            explanation_df["Contribution Score"].abs().sort_values(ascending=False).index
        )
        explanation_df = explanation_df.reset_index(drop=True)
        explanation_df["Contribution Score"] = explanation_df["Contribution Score"].round(3)
        st.dataframe(explanation_df, use_container_width=True)

        # --- Model Coefficient Analysis ---
        # Unlike LIME (which is local to this email), this shows which words
        # the model learned to associate with phishing across the whole training set —
        # a useful complementary view that's also much faster to compute
        st.subheader("Model Coefficient Analysis")

        coef_weights = explain_with_coefficients(model, vec, combined_text, top_n=10)
        coef_df = pd.DataFrame(coef_weights, columns=["Word or Phrase", "Contribution Score"])
        coef_df = coef_df.reset_index(drop=True)
        # Match the same rounding as the LIME table above
        coef_df["Contribution Score"] = coef_df["Contribution Score"].round(3)
        st.info(
            "**How to read this table:** Each row shows a word from the training data "
            "and its learned model weight. A **positive score** means that word is a "
            "strong indicator of **Phishing** emails in general. A **negative score** "
            "means that word is strongly associated with **Legitimate** emails. The "
            "larger the absolute value, the more influential that word is across the "
            "entire dataset — not just this specific email."
        )
        st.dataframe(coef_df, use_container_width=True)
        st.caption(
            "Positive scores indicate words associated with phishing across the entire "
            "training dataset. Negative scores indicate words associated with legitimate emails."
        )

        # Only log metadata — timestamp, label, confidence, text length.
        # The actual email content is never written to disk.
        ts = datetime.now().isoformat(timespec="seconds")
        log_line = f"{ts},{label},{confidence:.4f},{len(combined_text)}\n"
        if not LOG_FILE.exists():
            LOG_FILE.write_text("timestamp,prediction,confidence,text_length\n", encoding="utf-8")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(log_line)

        st.caption("Only metadata is logged locally (no email content stored).")
