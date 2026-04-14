"""app_streamlit.py — Streamlit web interface for the phishing email detector.

Run from the project root:
    streamlit run src/app_streamlit.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add src/ to the path so imports work from the project root or inside src/
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    CLASSIFICATION_REPORT_JSON,
    CONFUSION_MATRIX_PNG,
    DATASET_STATS_JSON,
    EVAL_DIR,
    LIME_NUM_FEATURES,
    LOG_DIR,
    MAX_INPUT_CHARS,
    MODEL_COMPARISON_CSV,
    ROC_CURVE_PNG,
)
from explain import build_explainer, explain_with_lime, explain_with_shap
from predict import load_assets, predict_email

# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------
import requests


def download_assets() -> None:
    """Download model and evaluation files from GitHub releases if they don't already exist locally."""
    _base = "https://github.com/hsalim3113/phishguard/releases/download/v1.0"
    files = {
        Path("models/model_logreg.joblib"): f"{_base}/model_logreg.joblib",
        Path("models/tfidf_vectorizer.joblib"): f"{_base}/tfidf_vectorizer.joblib",
        Path("outputs/evaluation/classification_report.json"): f"{_base}/classification_report.json",
        Path("outputs/evaluation/confusion_matrix.png"): f"{_base}/confusion_matrix.png",
        Path("outputs/evaluation/roc_curve.png"): f"{_base}/roc_curve.png",
        Path("outputs/evaluation/model_comparison.png"): f"{_base}/model_comparison.png",
        Path("outputs/evaluation/dataset_stats.json"): f"{_base}/dataset_stats.json",
    }

    for dest, url in files.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            response = requests.get(url)
            dest.write_bytes(response.content)


try:
    download_assets()
except Exception as e:
    st.warning(f"Some assets could not be downloaded: {e}")

# ---------------------------------------------------------------------------
# App config and startup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions_log.csv"


# cache_resource loads the model once at startup rather than on every button click
@st.cache_resource
def get_assets():
    """Load model, vectoriser, and LIME explainer (cached across sessions).

    Returns:
        tuple: (model, vec, explainer) or raises SystemExit on missing files.
    """
    model, vec = load_assets()
    explainer = build_explainer()
    return model, vec, explainer


try:
    model, vec, explainer = get_assets()
except Exception:
    st.error(
        "**Model files not found.** Please run `python src/train.py` first to "
        "generate the model."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar, shows the evaluation metrics saved by train.py
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Performance")
    with st.expander("View evaluation results", expanded=False):
        report_path = CLASSIFICATION_REPORT_JSON
        if report_path.exists():
            try:
                report = json.loads(report_path.read_bytes().decode("utf-8-sig"))
            except Exception:
                report = None
        else:
            report = None
        if report is not None:
            rows = []
            for cls in ["Legitimate", "Phishing", "weighted avg"]:
                key = cls.lower() if cls == "Legitimate" else cls
                # the key casing in classification_report can vary so match it case-insensitively
                matched = next((v for k, v in report.items()
                                if k.lower() == cls.lower()), None)
                if matched and isinstance(matched, dict):
                    rows.append({
                        "Class": cls,
                        "Precision": f"{matched.get('precision', 0):.3f}",
                        "Recall": f"{matched.get('recall', 0):.3f}",
                        "F1": f"{matched.get('f1-score', 0):.3f}",
                        "Support": int(matched.get("support", 0)),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            acc = report.get("accuracy")
            if acc:
                st.metric("Overall accuracy", f"{acc:.3f}")

        if CONFUSION_MATRIX_PNG.exists():
            st.image(str(CONFUSION_MATRIX_PNG), caption="Confusion Matrix", use_container_width=True)

        if ROC_CURVE_PNG.exists():
            st.image(str(ROC_CURVE_PNG), caption="ROC Curve", use_container_width=True)

    st.divider()
    st.caption("Phishing Detector Prototype — educational use only")


# ---------------------------------------------------------------------------
# Main page, title and info sections
# ---------------------------------------------------------------------------
st.title("🛡️ Phishing Email Detector")
st.write("Paste an email subject and body. The system predicts whether it is "
         "phishing or legitimate and explains the key contributing words.")

with st.expander("About this model"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Model:** Logistic Regression  \n"
            "**Vectoriser:** TF-IDF (max 30,000 features, bigrams)  \n"
            "**Explainability:** LIME (primary), SHAP (secondary)"
        )
    with col2:
        stats_path = DATASET_STATS_JSON
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_bytes().decode("utf-8-sig"))
            except Exception:
                stats = None
        else:
            stats = None
        if stats is not None:
            st.markdown(
                f"**Training samples:** {stats.get('total_samples', 'N/A'):,}  \n"
                f"**Legitimate:** {stats.get('n_legitimate', 'N/A'):,} "
                f"({stats.get('pct_legitimate', '?')}%)  \n"
                f"**Phishing:** {stats.get('n_phishing', 'N/A'):,} "
                f"({stats.get('pct_phishing', '?')}%)"
            )

        comp_path = MODEL_COMPARISON_CSV
        if comp_path.exists():
            st.markdown(f"[View model comparison results]({comp_path})")

with st.expander("How it works"):
    st.markdown("""
**TF-IDF (Term Frequency–Inverse Document Frequency)**
Converts email text into a numerical vector. Words that appear frequently in
one email but rarely across all emails receive a higher score — these are the
distinctive "signal" words. Common words like *"the"* or *"and"* are
down-weighted automatically.

**Logistic Regression**
Takes the TF-IDF vector and outputs a probability between 0 and 1. Values
closer to 1 indicate phishing; closer to 0 indicate legitimate. The model
learns which word patterns are statistically associated with phishing during
training.

**Confidence score**
The raw probability produced by the model. A score of 0.95 means the model is
95% confident in its prediction. Scores near 0.5 indicate uncertainty.

**LIME explanations**
LIME perturbs the input by randomly masking words and observing how the
prediction changes. Words whose removal shifts the prediction strongly are
ranked as the most influential.

**SHAP explanations**
SHAP uses the model's own coefficients to assign each word a contribution
score. For Logistic Regression the values are mathematically exact (no
sampling needed).

**Learning mode**
Guess before seeing the prediction. Your session score tracks how many you get
right — useful for training your own intuition about phishing language.
    """)

st.divider()

# ---------------------------------------------------------------------------
# Tabs — split the app into the email analyser and the training mode
# using st.tabs so both sections are always visible in the nav bar at the top
# ---------------------------------------------------------------------------
tab_analyser, tab_training = st.tabs(["📧 Email Analyser", "🎯 Training Mode"])


# ===========================================================================
# TAB 1 — Email Analyser (unchanged from original)
# ===========================================================================
with tab_analyser:

    # -------------------------------------------------------------------------
    # User input — subject and body with quick-load example buttons
    # -------------------------------------------------------------------------
    if "example_subject" not in st.session_state:
        st.session_state.example_subject = ""
    if "example_body" not in st.session_state:
        st.session_state.example_body = ""

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("Load phishing example"):
            st.session_state.example_subject = "Urgent: Your account has been suspended"
            st.session_state.example_body = (
                "Dear customer, your account has been suspended due to suspicious activity. "
                "Please verify your details immediately by clicking the link below or your "
                "account will be permanently closed. "
                "Click here: http://secure-login-verify.com/account"
            )
            st.rerun()
    with col_ex2:
        if st.button("Load legitimate example"):
            st.session_state.example_subject = "Team meeting rescheduled to Thursday"
            st.session_state.example_body = (
                "Hi everyone, just a quick note to let you know the weekly team meeting "
                "has been moved from Wednesday to Thursday at 2pm. Same room as usual. "
                "Please update your calendars accordingly. Thanks, Sarah"
            )
            st.rerun()

    subject = st.text_input("Email subject", value=st.session_state.example_subject)
    body = st.text_area("Email body", height=220, value=st.session_state.example_body)

    char_count = len(subject) + len(body)
    if char_count > 0:
        st.caption(f"Characters: {char_count:,} / {MAX_INPUT_CHARS:,}")

    explain_method = st.radio(
        "Explanation method",
        ["LIME", "SHAP"],
        horizontal=True,
        help="LIME uses random perturbation; SHAP uses model coefficients directly.",
    )

    # -------------------------------------------------------------------------
    # Learning mode
    # -------------------------------------------------------------------------
    # session_state keeps values between button clicks so we can track the score
    if "score_total" not in st.session_state:
        st.session_state.score_total = 0
    if "score_correct" not in st.session_state:
        st.session_state.score_correct = 0
    if "streak" not in st.session_state:
        st.session_state.streak = 0

    learning_mode = st.checkbox("Learning mode (guess first)")
    user_guess = None
    if learning_mode:
        user_guess = st.radio("Your guess", ["phishing", "legitimate"], horizontal=True)
        total = st.session_state.score_total
        correct = st.session_state.score_correct
        streak = st.session_state.streak
        pct = round(correct / total * 100) if total > 0 else 0
        st.info(
            f"Your score: **{correct}/{total} correct ({pct}%)** | "
            f"Current streak: **{streak}**"
        )
        if st.button("Reset score"):
            st.session_state.score_total = 0
            st.session_state.score_correct = 0
            st.session_state.streak = 0
            st.rerun()

    # -------------------------------------------------------------------------
    # Prediction and explanation
    # -------------------------------------------------------------------------
    if st.button("Analyse email", type="primary"):

        combined_raw = (subject or "").strip() + " " + (body or "").strip()
        if not combined_raw.strip():
            st.error("Please enter a subject or a body before analysing.")
            st.stop()

        if len(combined_raw.strip()) < 20:
            st.warning(
                "Input is very short — results may be unreliable. "
                "Try adding more context."
            )

        if len(combined_raw) > MAX_INPUT_CHARS:
            st.warning(
                f"Email is very long — only the first {MAX_INPUT_CHARS:,} "
                "characters will be analysed."
            )

        try:
            label, confidence, combined_text = predict_email(model, vec, subject, body)
        except Exception as exc:
            st.error(f"Prediction failed. Please check your input and try again.  \n`{exc}`")
            st.stop()

        st.subheader("Result")
        col_res, col_conf = st.columns(2)
        with col_res:
            if label == "phishing":
                st.error(f"Prediction: **PHISHING** 🚨")
            else:
                st.success(f"Prediction: **LEGITIMATE** ✅")
        with col_conf:
            st.metric("Confidence", f"{confidence:.1%}")

        if learning_mode and user_guess is not None:
            st.subheader("Learning mode feedback")
            is_correct = user_guess == label
            st.session_state.score_total += 1
            if is_correct:
                st.session_state.score_correct += 1
                st.session_state.streak += 1
                st.success(f"Correct! You guessed **{user_guess}** — the model agrees.")
            else:
                st.session_state.streak = 0
                st.warning(
                    f"Incorrect. You guessed **{user_guess}** but the model predicted "
                    f"**{label}**."
                )

        st.subheader(f"Explanation — top contributing words ({explain_method})")

        weights = []
        try:
            if explain_method == "LIME":
                weights = explain_with_lime(explainer, model, vec, combined_text,
                                            num_features=LIME_NUM_FEATURES)
            else:
                weights = explain_with_shap(model, vec, combined_text)

            if not weights:
                raise ValueError("Explainer returned no features.")

        except Exception as exc:
            st.warning(
                "Explanation could not be generated for this input — "
                f"try a longer email.  \n`{exc}`"
            )
            weights = []

        if weights:
            words = [w[0] for w in weights]
            values = [w[1] for w in weights]
            # Red for words that push toward phishing, green for legitimate
            colours = ["#d62728" if v > 0 else "#2ca02c" for v in values]

            fig, ax = plt.subplots(figsize=(7, max(3, len(weights) * 0.45)))
            bars = ax.barh(words[::-1], values[::-1], color=colours[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Weight (positive = phishing, negative = legitimate)")
            ax.set_title(f"{explain_method} Feature Weights")
            for bar, val in zip(bars, values[::-1]):
                x_pos = bar.get_width() + (0.001 if val >= 0 else -0.001)
                ha = "left" if val >= 0 else "right"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center", ha=ha, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "🔴 **Red bars** = words pushing toward *phishing* classification  \n"
                "🟢 **Green bars** = words pushing toward *legitimate* classification  \n"
                "**Longer bars** = stronger influence on the prediction"
            )

            # In learning mode, show which words actually influenced the result
            if learning_mode:
                top3 = [w[0] for w in sorted(weights, key=lambda x: abs(x[1]),
                                              reverse=True)[:3]]
                if label == "phishing":
                    st.info(
                        f"This email was flagged mainly because of words like: "
                        f"**{', '.join(top3)}**. These patterns are common in phishing "
                        "emails because they often create urgency, impersonate trusted "
                        "entities, or contain unusual terminology."
                    )
                else:
                    st.info(
                        f"This email appeared legitimate. Key indicators were: "
                        f"**{', '.join(top3)}**. These words are more typical of "
                        "normal correspondence than phishing attempts."
                    )

        # Only log metadata, not the actual email content
        ts = datetime.now().isoformat(timespec="seconds")
        log_line = f"{ts},{label},{confidence:.4f},{len(combined_text)}\n"
        if not LOG_FILE.exists():
            LOG_FILE.write_text("timestamp,prediction,confidence,text_length\n",
                                encoding="utf-8")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(log_line)

        st.caption("Only metadata is logged locally — no email content is stored.")


# ===========================================================================
# TAB 2 — Training Mode
# Walk the user through a fixed set of examples one at a time, ask them to
# guess, then run the real model and show LIME so they can see why it decided
# what it did. Basically an interactive quiz using the actual classifier.
# ===========================================================================
with tab_training:

    # -------------------------------------------------------------------------
    # Fixed list of training examples — mix of phishing and legitimate.
    # I kept these varied on purpose so the model has to deal with different
    # styles of phishing (urgency, prizes, IT impersonation) and different
    # styles of normal email (workplace, personal, transactional).
    # -------------------------------------------------------------------------
    TRAINING_EXAMPLES = [
        {
            "subject": "Action required: Verify your PayPal account now",
            "body": (
                "We have noticed unusual sign-in activity on your PayPal account. "
                "Your account access has been limited. To restore full access, please "
                "confirm your identity by clicking the link below within 24 hours. "
                "Failure to verify will result in permanent account suspension. "
                "Verify now: http://paypal-secure-verify.net/confirm"
            ),
            "label": "phishing",
        },
        {
            "subject": "Lunch catch-up this week?",
            "body": (
                "Hey, hope you're well! I'll be in the city on Friday and wondered if "
                "you fancied grabbing lunch? There's that new place on George Street "
                "that opened last month — supposed to be pretty good. Let me know if "
                "you're free, maybe around 12:30? No worries if not, we can sort "
                "something another time. Cheers, Tom"
            ),
            "label": "legitimate",
        },
        {
            "subject": "Congratulations! You have been selected as a prize winner",
            "body": (
                "Dear valued customer, you have been randomly selected to receive a "
                "£500 gift card as part of our annual loyalty reward programme. "
                "To claim your prize you must provide your full name, address, and "
                "bank details within 48 hours. Click here to claim: "
                "http://rewards-claim-now.com/prize"
            ),
            "label": "phishing",
        },
        {
            "subject": "Your order has been dispatched — estimated delivery Thursday",
            "body": (
                "Hi, your order #38471 has been picked and dispatched from our "
                "warehouse. Your parcel is now with Royal Mail and should arrive by "
                "Thursday 3rd April. You can track your delivery using the reference "
                "number RM004731821GB on the Royal Mail website. If you have any "
                "questions about your order please reply to this email. Thanks for "
                "shopping with us."
            ),
            "label": "legitimate",
        },
        {
            "subject": "IT Department: Mandatory password reset — expires today",
            "body": (
                "All staff are required to reset their network password before end of "
                "business today following a scheduled security audit. Please use the "
                "link below to reset your credentials immediately. If you do not reset "
                "your password today your account will be locked and you will need to "
                "contact the helpdesk in person. Reset here: "
                "http://company-it-reset.xyz/staff-portal"
            ),
            "label": "phishing",
        },
        {
            "subject": "Notes from today's project meeting",
            "body": (
                "Hi all, here are the action points from this afternoon's catch-up. "
                "James is looking into the API timeout issue and will report back by "
                "Wednesday. Priya is updating the requirements doc and wants everyone "
                "to review it before the next sprint. We also agreed to push the demo "
                "back one week to give the team more time for testing. Next meeting "
                "same time next Tuesday. Let me know if I missed anything. — Rachel"
            ),
            "label": "legitimate",
        },
        {
            "subject": "Your Netflix account will be cancelled unless you update billing",
            "body": (
                "We were unable to process your most recent payment. As a result your "
                "Netflix membership is at risk of cancellation. Please update your "
                "payment information immediately to avoid any interruption to your "
                "service. Your account will be closed in 12 hours if no action is "
                "taken. Update billing: http://netflix-billing-update.info/secure"
            ),
            "label": "phishing",
        },
        {
            "subject": "Re: dissertation feedback — chapter 3",
            "body": (
                "Hi, I've had a chance to read through chapter 3 now. Overall it's "
                "coming along well — the literature review section is much stronger "
                "than the previous draft. Main thing I'd flag is that the methodology "
                "section needs a bit more justification for why you chose a qualitative "
                "approach over a mixed methods one. Happy to chat through it if that "
                "would help. Drop me a line and we can find a time. Best, Dr. Hughes"
            ),
            "label": "legitimate",
        },
    ]

    # total number of examples in the exercise
    TOTAL_EXAMPLES = len(TRAINING_EXAMPLES)

    # -------------------------------------------------------------------------
    # Session state for the training mode — I'm initialising all of these here
    # at the top so I can reference them anywhere below without key errors.
    # tm_index tracks which example we're on (0-based).
    # tm_submitted flips to True when the user clicks "Submit guess" so we know
    # to show the result instead of the guess form.
    # tm_result_* store the model output so it survives the rerun that happens
    # when the user clicks Next.
    # -------------------------------------------------------------------------
    if "tm_index" not in st.session_state:
        st.session_state.tm_index = 0
    if "tm_score" not in st.session_state:
        st.session_state.tm_score = 0
    if "tm_submitted" not in st.session_state:
        st.session_state.tm_submitted = False
    if "tm_finished" not in st.session_state:
        st.session_state.tm_finished = False
    # these hold the model result after submission so the chart stays visible
    if "tm_result_label" not in st.session_state:
        st.session_state.tm_result_label = None
    if "tm_result_confidence" not in st.session_state:
        st.session_state.tm_result_confidence = None
    if "tm_result_weights" not in st.session_state:
        st.session_state.tm_result_weights = []
    if "tm_user_guess" not in st.session_state:
        st.session_state.tm_user_guess = None

    # -------------------------------------------------------------------------
    # Restart button — always visible at the top so people can reset without
    # having to scroll to the end of a finished run
    # -------------------------------------------------------------------------
    if st.button("Restart training exercise", key="tm_restart"):
        st.session_state.tm_index = 0
        st.session_state.tm_score = 0
        st.session_state.tm_submitted = False
        st.session_state.tm_finished = False
        st.session_state.tm_result_label = None
        st.session_state.tm_result_confidence = None
        st.session_state.tm_result_weights = []
        st.session_state.tm_user_guess = None
        st.rerun()

    st.divider()

    # -------------------------------------------------------------------------
    # Summary / end screen — shown once the user has gone through all examples.
    # I calculate a percentage and pick a message based on how well they did.
    # The tip at the bottom is static but summarises the phishing patterns that
    # actually appeared across the examples above.
    # -------------------------------------------------------------------------
    if st.session_state.tm_finished:
        score = st.session_state.tm_score
        pct = round(score / TOTAL_EXAMPLES * 100)

        st.subheader("Exercise complete!")
        st.metric("Final score", f"{score} / {TOTAL_EXAMPLES} ({pct}%)")

        if pct == 100:
            st.success("Perfect score — you spotted everything!")
        elif pct >= 75:
            st.success("Good work, you caught most of them.")
        elif pct >= 50:
            st.warning("Not bad, but a few slipped through — worth reviewing the LIME charts.")
        else:
            st.error("Looks like phishing detection is tricky! Review the explanations below.")

        st.divider()
        st.subheader("Common phishing indicators in these examples")
        # these tips are based on the actual examples in TRAINING_EXAMPLES above
        st.markdown("""
**Urgency and deadlines** — phrases like *"expires today"*, *"within 24 hours"*, and
*"account will be closed"* are used to stop you thinking clearly before clicking.

**Suspicious URLs** — legitimate services use their own domain (e.g. `paypal.com`).
The phishing links above used domains like `paypal-secure-verify.net` and
`netflix-billing-update.info` — real-looking but wrong.

**Requests for sensitive information** — any email asking for bank details, passwords,
or personal info via a link should be treated with suspicion. Genuine companies rarely
do this by email.

**Impersonating known brands or internal IT** — the PayPal, Netflix, and IT department
examples all copy the tone and urgency of real communications but contain red flags in
the sender domain and link destination.

**Contrast with legitimate emails** — notice how the real emails (lunch invitation,
order dispatch, meeting notes) are specific, personal, and don't ask you to click
anything urgently or hand over information.
        """)

    # -------------------------------------------------------------------------
    # Main quiz loop — only shown while the exercise is still in progress
    # -------------------------------------------------------------------------
    else:
        idx = st.session_state.tm_index
        example = TRAINING_EXAMPLES[idx]

        # progress indicator so the user knows how far through they are
        st.caption(f"Example {idx + 1} of {TOTAL_EXAMPLES}  |  "
                   f"Score so far: {st.session_state.tm_score} / {idx}")
        st.progress((idx) / TOTAL_EXAMPLES)

        st.subheader(f"Example {idx + 1}")

        # show the email read-only — using text_input/text_area with disabled=True
        # so the fields look consistent with the analyser tab but can't be edited
        st.text_input("Subject", value=example["subject"], disabled=True,
                      key=f"tm_subj_{idx}")
        st.text_area("Body", value=example["body"], height=160, disabled=True,
                     key=f"tm_body_{idx}")

        st.divider()

        # ---------------------------------------------------------------------
        # Before submission — show the guess form
        # After submission — show the result and LIME chart
        # I'm using tm_submitted as the flag to switch between the two views.
        # Streamlit reruns the whole script on every interaction so I need the
        # results stored in session_state to still be there after clicking Next.
        # ---------------------------------------------------------------------
        if not st.session_state.tm_submitted:

            st.markdown("**Is this email phishing or legitimate?**")
            guess = st.radio(
                "Your guess",
                ["phishing", "legitimate"],
                horizontal=True,
                key=f"tm_guess_{idx}",
            )

            if st.button("Submit guess", type="primary", key=f"tm_submit_{idx}"):
                # run the real model on this example
                try:
                    label, confidence, combined_text = predict_email(
                        model, vec, example["subject"], example["body"]
                    )
                except Exception as exc:
                    st.error(f"Model prediction failed: {exc}")
                    st.stop()

                # run LIME on the combined text
                weights = []
                try:
                    weights = explain_with_lime(
                        explainer, model, vec, combined_text,
                        num_features=LIME_NUM_FEATURES,
                    )
                except Exception:
                    # LIME sometimes fails on very short inputs, that's fine
                    pass

                # store everything in session_state so it survives the rerun
                st.session_state.tm_user_guess = guess
                st.session_state.tm_result_label = label
                st.session_state.tm_result_confidence = confidence
                st.session_state.tm_result_weights = weights

                # update score — comparing against the ground truth label in
                # the example dict, not the model prediction, because this is
                # a quiz about whether the user can spot phishing, not whether
                # they agree with the model
                if guess == example["label"]:
                    st.session_state.tm_score += 1

                st.session_state.tm_submitted = True
                st.rerun()

        else:
            # -----------------------------------------------------------------
            # Result view — shown after the user has submitted their guess
            # -----------------------------------------------------------------
            guess = st.session_state.tm_user_guess
            label = st.session_state.tm_result_label
            confidence = st.session_state.tm_result_confidence
            weights = st.session_state.tm_result_weights
            ground_truth = example["label"]

            # was the user correct against the ground truth?
            user_correct = (guess == ground_truth)

            st.subheader("Result")

            col_truth, col_model, col_conf = st.columns(3)
            with col_truth:
                # show what the email actually was
                if ground_truth == "phishing":
                    st.error(f"Actual: **PHISHING** 🚨")
                else:
                    st.success(f"Actual: **LEGITIMATE** ✅")
            with col_model:
                # show what the model predicted — could differ from ground truth
                if label == "phishing":
                    st.error(f"Model: **PHISHING** 🚨")
                else:
                    st.success(f"Model: **LEGITIMATE** ✅")
            with col_conf:
                st.metric("Model confidence", f"{confidence:.1%}")

            # feedback on whether the user got it right
            if user_correct:
                st.success(f"You guessed **{guess}** — correct!")
            else:
                st.warning(
                    f"You guessed **{guess}** but this was a **{ground_truth}** email."
                )

            # if the model disagreed with the ground truth, flag it so the
            # user doesn't get confused — this does occasionally happen
            if label != ground_truth:
                st.info(
                    f"Note: the model predicted **{label}** but the correct answer "
                    f"is **{ground_truth}**. The model isn't always right."
                )

            # LIME bar chart — same rendering logic as the analyser tab
            if weights:
                st.subheader("LIME explanation — why did the model decide this?")

                words = [w[0] for w in weights]
                values = [w[1] for w in weights]
                colours = ["#d62728" if v > 0 else "#2ca02c" for v in values]

                fig, ax = plt.subplots(figsize=(7, max(3, len(weights) * 0.45)))
                bars = ax.barh(words[::-1], values[::-1], color=colours[::-1])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Weight (positive = phishing, negative = legitimate)")
                ax.set_title("LIME Feature Weights")
                for bar, val in zip(bars, values[::-1]):
                    x_pos = bar.get_width() + (0.001 if val >= 0 else -0.001)
                    ha = "left" if val >= 0 else "right"
                    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                            f"{val:+.3f}", va="center", ha=ha, fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "🔴 **Red bars** = words pushing toward *phishing* classification  \n"
                    "🟢 **Green bars** = words pushing toward *legitimate* classification  \n"
                    "**Longer bars** = stronger influence on the prediction"
                )
            else:
                st.info("LIME explanation not available for this example.")

            st.divider()

            # -----------------------------------------------------------------
            # Next button — advances to next example or triggers the end screen
            # tm_submitted gets reset to False so the guess form shows again
            # -----------------------------------------------------------------
            if idx + 1 >= TOTAL_EXAMPLES:
                next_label = "See results"
            else:
                next_label = f"Next example ({idx + 2} of {TOTAL_EXAMPLES})"

            if st.button(next_label, type="primary", key=f"tm_next_{idx}"):
                st.session_state.tm_submitted = False
                st.session_state.tm_result_label = None
                st.session_state.tm_result_confidence = None
                st.session_state.tm_result_weights = []
                st.session_state.tm_user_guess = None

                if idx + 1 >= TOTAL_EXAMPLES:
                    st.session_state.tm_finished = True
                else:
                    st.session_state.tm_index += 1

                st.rerun()
