import streamlit as st
from datetime import datetime
from pathlib import Path

from predict import load_assets, predict_email
from explain import build_explainer, explain_with_lime

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions_log.csv"

st.set_page_config(page_title="Phishing Detector Prototype", layout="centered")

st.title("Phishing Email Detector (Prototype)")
st.write("Paste an email subject and body. The system returns a prediction and a simple explanation.")

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
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")

    if learning_mode and user_guess is not None:
        st.subheader("Learning mode feedback")
        st.write(f"Your guess: {user_guess}")
        st.write("Match" if user_guess == label else "Mismatch")

    st.subheader("Explanation (top contributing words)")
    weights = explain_with_lime(explainer, model, vec, combined_text, num_features=10)
    st.write(weights)

    ts = datetime.now().isoformat(timespec="seconds")
    log_line = f"{ts},{label},{confidence:.4f},{len(combined_text)}\n"
    if not LOG_FILE.exists():
        LOG_FILE.write_text("timestamp,prediction,confidence,text_length\n", encoding="utf-8")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(log_line)

    st.caption("Only metadata is logged locally (no email content stored).")
