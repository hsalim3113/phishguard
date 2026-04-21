"""app_streamlit.py — Streamlit web interface for the phishing email detector.

Run from the project root:
    streamlit run src/app_streamlit.py
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# json: parses classification report and dataset stats JSON files written by train.py.
import json

# sys: modifies the module search path so local modules (config, explain, predict) import correctly.
import sys

# datetime: records UTC timestamps for Firestore; timezone.utc keeps them timezone-aware.
from datetime import datetime, timezone

# pathlib.Path: checks whether local model and evaluation files exist.
from pathlib import Path

# matplotlib: renders LIME/SHAP bar charts. "Agg" backend must be set before pyplot
# import so charts render headlessly on servers without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pandas: builds the metrics DataFrame for the sidebar Model Performance table.
import pandas as pd

# requests: used for Firebase Authentication REST API calls and GitHub Release downloads.
import requests

# streamlit: the web framework — every st.* call renders a UI element in the browser.
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure local modules are importable
# ---------------------------------------------------------------------------

# Add src/ to sys.path so "import config" finds src/config.py regardless of launch directory.
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    CLASSIFICATION_REPORT_JSON,
    CONFUSION_MATRIX_PNG,
    DATASET_STATS_JSON,
    LIME_NUM_FEATURES,
    LOG_DIR,
    MAX_INPUT_CHARS,
    MODEL_COMPARISON_CSV,
    ROC_CURVE_PNG,
)

from explain import build_explainer, explain_with_lime, explain_with_shap, explain_with_shap_xgb
from predict import load_assets, predict_email


# ---------------------------------------------------------------------------
# Asset download — fetches model files from GitHub Releases at first startup
# ---------------------------------------------------------------------------

def download_assets() -> None:
    """Download model and evaluation files from GitHub Releases if they are missing locally.

    Checks dest.exists() before each download so files are only fetched once —
    without this check every page interaction would re-download hundreds of MB.
    """
    _base = "https://github.com/hsalim3113/phishguard/releases/download/v1.0"

    files = {
        Path("models/model_logreg.joblib"): f"{_base}/model_logreg.joblib",
        Path("models/model_xgb.joblib"): f"{_base}/model_xgb.joblib",
        Path("models/tfidf_vectorizer.joblib"): f"{_base}/tfidf_vectorizer.joblib",
        Path("outputs/evaluation/classification_report.json"): f"{_base}/classification_report.json",
        Path("outputs/evaluation/confusion_matrix.png"): f"{_base}/confusion_matrix.png",
        Path("outputs/evaluation/roc_curve.png"): f"{_base}/roc_curve.png",
        Path("outputs/evaluation/model_comparison.png"): f"{_base}/model_comparison.png",
        Path("outputs/evaluation/dataset_stats.json"): f"{_base}/dataset_stats.json",
    }

    for dest, url in files.items():
        # parents=True creates intermediate dirs; exist_ok=True avoids errors on re-runs.
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            response = requests.get(url)
            # write_bytes preserves binary content — write_text would corrupt .joblib and .png files.
            dest.write_bytes(response.content)


# Run once at module startup; try/except shows a non-fatal warning if download partially fails.
try:
    download_assets()
except Exception as e:
    st.warning(f"Some assets could not be downloaded: {e}")


# ---------------------------------------------------------------------------
# Firebase Authentication — REST API helpers
# ---------------------------------------------------------------------------
# Uses the Firebase Identity Toolkit REST API directly (no pyrebase4) to avoid
# dependency conflicts. All operations POST JSON to endpoint URLs with the API key.

_FIREBASE_AUTH_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"


def _api_key() -> str:
    """Return the Firebase Web API key from Streamlit secrets.

    Stored in .streamlit/secrets.toml under [firebase][apiKey] to keep it out of source control.
    """
    return st.secrets["firebase"]["apiKey"]


def _firebase_post(endpoint: str, payload: dict) -> dict:
    """Send a POST request to a Firebase Identity Toolkit endpoint and return the JSON response.

    Args:
        endpoint (str): Firebase operation name, e.g. "signInWithPassword".
        payload (dict): JSON request body with credentials or idToken.

    Returns:
        dict: Parsed JSON response from Firebase on success.

    Raises:
        Exception: With the Firebase error message string if HTTP status is not OK.
    """
    url = f"{_FIREBASE_AUTH_BASE}:{endpoint}?key={_api_key()}"

    # timeout=10 prevents the app from hanging indefinitely on network failure.
    resp = requests.post(url, json=payload, timeout=10)
    data = resp.json()

    if not resp.ok:
        error_msg = data.get("error", {}).get("message", "UNKNOWN_ERROR")
        raise Exception(error_msg)

    return data


def firebase_sign_in(email: str, password: str) -> dict:
    """Sign in an existing Firebase user with email and password.

    Returns:
        dict: Firebase response containing idToken, localId, and account metadata.
    """
    # returnSecureToken: True instructs Firebase to include the ID token in the response.
    return _firebase_post(
        "signInWithPassword",
        {"email": email, "password": password, "returnSecureToken": True},
    )


def firebase_register(email: str, password: str) -> dict:
    """Create a new Firebase Auth account with email and password.

    Returns:
        dict: Same structure as firebase_sign_in — new account's idToken and localId.
    """
    return _firebase_post(
        "signUp",
        {"email": email, "password": password, "returnSecureToken": True},
    )


def firebase_get_account_info(id_token: str) -> dict:
    """Fetch the Firebase account profile for the currently signed-in user.

    The sign-in response omits displayName — this "lookup" call retrieves the full profile.

    Args:
        id_token (str): Firebase ID token from firebase_sign_in() or firebase_register().

    Returns:
        dict: Firebase response with a "users" list containing profile fields.
    """
    return _firebase_post("lookup", {"idToken": id_token})


def firebase_update_profile(id_token: str, display_name: str) -> dict:
    """Set the display name on a newly created Firebase account.

    signUp doesn't accept displayName — a separate "update" call is required.

    Args:
        id_token (str): ID token of the newly created account.
        display_name (str): Name entered in the registration form.

    Returns:
        dict: Firebase response confirming the profile update.
    """
    return _firebase_post(
        "update",
        {"idToken": id_token, "displayName": display_name, "returnSecureToken": True},
    )


@st.cache_resource
def _firestore_client():
    """Initialise and return an authenticated Firestore database client.

    Returns None if GCP credentials are missing — callers check for None and skip
    Firestore operations, so the rest of the app works without Firestore configured.
    """
    try:
        from google.oauth2 import service_account
        from google.cloud import firestore

        # dict() converts the TOML AttrDict from secrets to a plain Python dict.
        creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return firestore.Client(credentials=credentials, project=creds_dict["project_id"])
    except Exception:
        return None


def _auth_error_message(exc: Exception) -> str:
    """Convert a raw Firebase error code into a user-friendly plain-English message.

    Args:
        exc (Exception): Exception raised by _firebase_post(); str() gives the error code.

    Returns:
        str: Plain-English message suitable for st.error().
    """
    msg = str(exc)

    if "INVALID_PASSWORD" in msg or "EMAIL_NOT_FOUND" in msg or "INVALID_LOGIN_CREDENTIALS" in msg:
        return "Invalid email or password."
    if "EMAIL_EXISTS" in msg:
        return "An account with this email already exists."
    if "WEAK_PASSWORD" in msg:
        return "Password must be at least 6 characters."
    if "INVALID_EMAIL" in msg:
        return "Please enter a valid email address."
    return f"Authentication error: {exc}"


def show_auth_page() -> None:
    """Render the login / register page and block access to the rest of the app.

    The caller calls st.stop() immediately after this returns, so the main app
    is completely invisible to unauthenticated users.
    """
    st.title("🛡️ PhishGuard")
    st.write("Sign in to access the phishing detector.")

    # Two tabs so the user can switch between logging in and creating a new account.
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", key="btn_login"):
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                try:
                    user = firebase_sign_in(email, password)

                    # The sign-in response omits displayName — fetch it separately.
                    info = firebase_get_account_info(user["idToken"])
                    display_name = info["users"][0].get("displayName") or email.split("@")[0]

                    # Store user details in session_state — persists across Streamlit reruns.
                    st.session_state.user = {
                        "uid": user["localId"],
                        "email": email,
                        "display_name": display_name,
                        "id_token": user["idToken"],
                    }

                    # Rerun so the auth gate passes and the main app renders immediately.
                    st.rerun()
                except Exception as exc:
                    st.error(_auth_error_message(exc))

    with tab_register:
        display_name = st.text_input("Display name", key="reg_name")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_password")

        if st.button("Create account", type="primary", key="btn_register"):
            if not display_name or not email or not password:
                st.error("Please fill in all fields.")
            else:
                try:
                    # Step 1: create account — Step 2: set display name (separate call).
                    user = firebase_register(email, password)
                    firebase_update_profile(user["idToken"], display_name=display_name)

                    st.session_state.user = {
                        "uid": user["localId"],
                        "email": email,
                        "display_name": display_name,
                        "id_token": user["idToken"],
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(_auth_error_message(exc))


# ---------------------------------------------------------------------------
# Firestore operations — persist and retrieve user data
# ---------------------------------------------------------------------------

def save_prediction(uid: str, subject: str, label: str, confidence: float,
                    top_features: list) -> None:
    """Save a single prediction result to the user's Firestore sub-collection.

    Stored at /users/{uid}/predictions/{auto-id}. Only metadata is saved (no email
    content) — full body text is excluded for privacy. Returns silently if Firestore
    is not configured.
    """
    db = _firestore_client()
    if db is None:
        return

    try:
        db.collection("users").document(uid).collection("predictions").add({
            "timestamp": datetime.now(timezone.utc),
            "subject": subject[:200],
            "label": label,
            "confidence": round(confidence, 4),
            # Top 3 words — enough for the history panel summary.
            "top_features": top_features[:3],
        })
    except Exception:
        pass  # Non-critical — prediction result is still shown to the user


def get_history(uid: str) -> list:
    """Retrieve the user's most recent 10 predictions from Firestore.

    Returns:
        list[dict]: Up to 10 prediction dicts in descending timestamp order.
            Returns an empty list if Firestore is unavailable.
    """
    db = _firestore_client()
    if db is None:
        return []
    try:
        docs = (
            db.collection("users").document(uid).collection("predictions")
            .order_by("timestamp", direction="DESCENDING")
            .limit(10)
            .stream()
        )
        return [doc.to_dict() for doc in docs]
    except Exception:
        return []


def save_quiz_score(uid: str, score: int, total: int) -> None:
    """Save the quiz score to Firestore, overwriting only if the new score is a personal best.

    Reads the current best before writing — merge=True preserves other fields on the
    user document that might exist alongside the score fields.
    """
    db = _firestore_client()
    if db is None:
        return
    try:
        ref = db.collection("users").document(uid)
        doc = ref.get()
        current_best = doc.to_dict().get("quiz_best_score", 0) if doc.exists else 0

        if score > current_best:
            ref.set({"quiz_best_score": score, "quiz_total": total}, merge=True)
    except Exception:
        pass


def get_best_quiz_score(uid: str) -> int:
    """Return the user's all-time best quiz score from Firestore.

    Returns:
        int: Highest score ever recorded for this user, or 0 if unavailable.
    """
    db = _firestore_client()
    if db is None:
        return 0
    try:
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict().get("quiz_best_score", 0)
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# App configuration and startup
# ---------------------------------------------------------------------------

# set_page_config MUST be the first Streamlit call — any earlier st.* call raises an error.
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
)

# Create the log directory and define the log file path for local audit trail.
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions_log.csv"


@st.cache_resource
def get_assets():
    """Load the Logistic Regression model, TF-IDF vectoriser, and LIME explainer from disk.

    @st.cache_resource runs this once per server session — without caching the ~50MB
    model would reload from disk on every button click, causing 1–2 second delays.

    Returns:
        tuple: (model, vec, explainer) needed for prediction and LIME explanation.

    Raises:
        FileNotFoundError: If joblib files don't exist (train.py not yet run).
    """
    model, vec = load_assets()
    explainer = build_explainer()
    return model, vec, explainer


@st.cache_resource
def get_xgb_model():
    """Load the XGBoost model from disk (cached across sessions).

    Separated from get_assets() because XGBoost is only needed for SHAP (XGBoost)
    and may be absent if train.py was run before XGBoost was added.

    Returns:
        XGBClassifier if models/model_xgb.joblib exists, None otherwise.
    """
    from joblib import load
    xgb_path = Path("models/model_xgb.joblib")
    if xgb_path.exists():
        return load(xgb_path)
    return None


# Load the primary model at startup — show a clear error and halt if files are missing.
try:
    model, vec, explainer = get_assets()
except Exception:
    st.error(
        "**Model files not found.** Please run `python src/train.py` first to "
        "generate the model."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Authentication gate
# ---------------------------------------------------------------------------

# Initialise on first load — without this the "is None" check below raises KeyError.
if "user" not in st.session_state:
    st.session_state.user = None

# st.stop() is critical — it prevents ALL subsequent code from running for unauthenticated users.
if st.session_state.user is None:
    show_auth_page()
    st.stop()

# Shortcut so the rest of the file reads _user["uid"] instead of st.session_state.user["uid"].
_user = st.session_state.user


# ---------------------------------------------------------------------------
# Sidebar — user info, logout, model performance, prediction history
# ---------------------------------------------------------------------------
with st.sidebar:

    # "  \n" is the Streamlit markdown convention for a line break within a markdown string.
    st.markdown(f"**{_user['display_name']}**  \n{_user['email']}")

    # Logout: sets user to None so the auth gate shows the login page on next rerun.
    if st.button("Logout", key="btn_logout"):
        st.session_state.user = None
        st.rerun()

    st.divider()

    st.header("Model Performance")

    with st.expander("View evaluation results", expanded=False):

        # .decode("utf-8-sig") strips the BOM that Windows tools sometimes add to UTF-8 files.
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

    st.subheader("My History")

    with st.expander("Last 10 predictions", expanded=False):
        history = get_history(_user["uid"])
        if history:
            for entry in history:
                ts = entry.get("timestamp")
                if hasattr(ts, "strftime"):
                    ts_str = ts.strftime("%d %b %H:%M")
                else:
                    ts_str = str(ts)[:16]

                label_h = entry.get("label", "?")
                conf_h = entry.get("confidence", 0)
                subj_h = (entry.get("subject") or "")[:40]
                icon = "🚨" if label_h == "phishing" else "✅"
                st.caption(f"{icon} **{label_h}** {conf_h:.0%} — {ts_str}  \n*{subj_h}*")
        else:
            st.caption("No predictions yet.")

    st.divider()
    st.caption("Phishing Detector Prototype — educational use only")


# ---------------------------------------------------------------------------
# Main page — title and information expanders
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
        # utf-8-sig handles any byte-order mark Windows tools may add.
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
# Tab navigation — Email Analyser and Training Mode
# ---------------------------------------------------------------------------
tab_analyser, tab_training = st.tabs(["📧 Email Analyser", "🎯 Training Mode"])


# ===========================================================================
# TAB 1 — Email Analyser
# ===========================================================================
with tab_analyser:

    # ---------------------------------------------------------------------------
    # Example email buttons
    # ---------------------------------------------------------------------------
    # session_state stores pre-filled text between reruns — the correct Streamlit
    # pattern for dynamically updating input field values.
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

    # ---------------------------------------------------------------------------
    # Email input fields
    # ---------------------------------------------------------------------------

    # value= reads session_state so example buttons pre-fill the fields.
    subject = st.text_input("Email subject", value=st.session_state.example_subject)
    body = st.text_area("Email body", height=220, value=st.session_state.example_body)

    # Live character counter — lets the user know if their input will be truncated.
    char_count = len(subject) + len(body)
    if char_count > 0:
        st.caption(f"Characters: {char_count:,} / {MAX_INPUT_CHARS:,}")

    # ---------------------------------------------------------------------------
    # Explanation method selector
    # ---------------------------------------------------------------------------
    explain_method = st.radio(
        "Explanation method",
        ["LIME", "SHAP (Logistic Regression)", "SHAP (XGBoost)"],
        horizontal=True,
        help=(
            "LIME uses random perturbation. "
            "SHAP (LogReg) uses LinearExplainer on model coefficients. "
            "SHAP (XGBoost) uses TreeExplainer on a gradient boosting model."
        ),
    )

    # ---------------------------------------------------------------------------
    # Learning mode
    # ---------------------------------------------------------------------------

    # Score and streak persist across multiple analyses within one session.
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

    # ---------------------------------------------------------------------------
    # Prediction and explanation
    # ---------------------------------------------------------------------------

    # All analysis code is inside this if block — nothing runs until the user clicks.
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

        # combined_text is the post-truncation text — passed to the explainer so the
        # explanation matches the prediction input exactly.
        try:
            label, confidence, combined_text = predict_email(model, vec, subject, body)
        except Exception as exc:
            st.error(f"Prediction failed. Please check your input and try again.  \n`{exc}`")
            st.stop()

        # ---------------------------------------------------------------------------
        # Display the prediction result
        # ---------------------------------------------------------------------------

        st.subheader("Result")

        col_res, col_conf = st.columns(2)
        with col_res:
            if label == "phishing":
                st.error(f"Prediction: **PHISHING** 🚨")
            else:
                st.success(f"Prediction: **LEGITIMATE** ✅")
        with col_conf:
            st.metric("Confidence", f"{confidence:.1%}")

        # ---------------------------------------------------------------------------
        # Learning mode feedback
        # ---------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------
        # Generate and display the explanation bar chart
        # ---------------------------------------------------------------------------

        st.subheader(f"Explanation — top contributing words ({explain_method})")

        weights = []
        try:
            if explain_method == "LIME":
                weights = explain_with_lime(explainer, model, vec, combined_text,
                                            num_features=LIME_NUM_FEATURES)

            elif explain_method == "SHAP (Logistic Regression)":
                weights = explain_with_shap(model, vec, combined_text)

            else:  # SHAP (XGBoost)
                xgb_model = get_xgb_model()
                if xgb_model is None:
                    raise ValueError(
                        "XGBoost model not found. Run `python src/train.py` first."
                    )
                weights = explain_with_shap_xgb(xgb_model, vec, combined_text)

            if not weights:
                raise ValueError("Explainer returned no features.")

        except Exception as exc:
            st.warning(
                "Explanation could not be generated for this input — "
                f"try a longer email.  \n`{exc}`"
            )
            weights = []

        # ---------------------------------------------------------------------------
        # Render the explanation bar chart
        # ---------------------------------------------------------------------------
        if weights:
            words = [w[0] for w in weights]
            values = [w[1] for w in weights]

            # Red = positive weight (toward phishing); green = negative (toward legitimate).
            colours = ["#d62728" if v > 0 else "#2ca02c" for v in values]

            # [::-1] reverses lists so the strongest contributor appears at the top.
            fig, ax = plt.subplots(figsize=(7, max(3, len(weights) * 0.45)))
            bars = ax.barh(words[::-1], values[::-1], color=colours[::-1])

            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Weight (positive = phishing, negative = legitimate)")
            ax.set_title(f"{explain_method} Feature Weights")

            # Annotate each bar with its numeric weight value just beyond the bar tip.
            for bar, val in zip(bars, values[::-1]):
                x_pos = bar.get_width() + (0.001 if val >= 0 else -0.001)
                ha = "left" if val >= 0 else "right"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center", ha=ha, fontsize=8)

            fig.tight_layout()
            st.pyplot(fig)

            # plt.close() releases figure memory — without it many analyses would accumulate RAM.
            plt.close(fig)

            st.caption(
                "🔴 **Red bars** = words pushing toward *phishing* classification  \n"
                "🟢 **Green bars** = words pushing toward *legitimate* classification  \n"
                "**Longer bars** = stronger influence on the prediction"
            )

            if explain_method == "SHAP (XGBoost)":
                st.info(
                    "**SHAP (XGBoost)** uses TreeExplainer on a gradient boosting model. "
                    "This is less directly interpretable than Logistic Regression (where each "
                    "weight maps to a single learned coefficient), but XGBoost typically "
                    "achieves marginally higher accuracy by combining many shallow decision "
                    "trees. The prediction above still uses Logistic Regression."
                )

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

        # ---------------------------------------------------------------------------
        # Persist the result to Firestore and the local log
        # ---------------------------------------------------------------------------

        # Top 3 words for the history panel summary — three words fit on one sidebar line.
        top_features = [w[0] for w in sorted(weights, key=lambda x: abs(x[1]),
                                              reverse=True)[:3]] if weights else []

        # save_prediction() returns silently if Firestore is not configured.
        save_prediction(_user["uid"], subject, label, confidence, top_features)

        # Local CSV log — lightweight audit trail that works without Firestore.
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
# ===========================================================================
with tab_training:

    # ---------------------------------------------------------------------------
    # Fixed list of training examples — 8 emails covering a range of scenarios.
    # Half phishing (urgency, prize claims, IT impersonation, brand spoofing);
    # half legitimate (personal, workplace, transactional, academic).
    # ---------------------------------------------------------------------------
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

    TOTAL_EXAMPLES = len(TRAINING_EXAMPLES)

    # ---------------------------------------------------------------------------
    # Training mode session state
    # ---------------------------------------------------------------------------
    # All quiz state lives in session_state — Streamlit reruns the entire script on
    # every interaction, so local variables would reset without it.
    if "tm_index" not in st.session_state:
        st.session_state.tm_index = 0
    if "tm_score" not in st.session_state:
        st.session_state.tm_score = 0
    if "tm_submitted" not in st.session_state:
        st.session_state.tm_submitted = False
    if "tm_finished" not in st.session_state:
        st.session_state.tm_finished = False
    if "tm_result_label" not in st.session_state:
        st.session_state.tm_result_label = None
    if "tm_result_confidence" not in st.session_state:
        st.session_state.tm_result_confidence = None
    if "tm_result_weights" not in st.session_state:
        st.session_state.tm_result_weights = []
    if "tm_user_guess" not in st.session_state:
        st.session_state.tm_user_guess = None
    if "tm_score_saved" not in st.session_state:
        st.session_state.tm_score_saved = False

    # ---------------------------------------------------------------------------
    # Restart button
    # ---------------------------------------------------------------------------
    if st.button("Restart training exercise", key="tm_restart"):
        st.session_state.tm_index = 0
        st.session_state.tm_score = 0
        st.session_state.tm_submitted = False
        st.session_state.tm_finished = False
        st.session_state.tm_result_label = None
        st.session_state.tm_result_confidence = None
        st.session_state.tm_result_weights = []
        st.session_state.tm_user_guess = None
        st.session_state.tm_score_saved = False
        st.rerun()

    st.divider()

    # ---------------------------------------------------------------------------
    # Finished screen
    # ---------------------------------------------------------------------------
    if st.session_state.tm_finished:
        score = st.session_state.tm_score
        pct = round(score / TOTAL_EXAMPLES * 100)

        # tm_score_saved prevents re-saving on every rerun while the finished screen is visible.
        if not st.session_state.tm_score_saved:
            save_quiz_score(_user["uid"], score, TOTAL_EXAMPLES)
            st.session_state.tm_score_saved = True

        best = get_best_quiz_score(_user["uid"])

        st.subheader("Exercise complete!")

        col_score, col_best = st.columns(2)
        with col_score:
            st.metric("Final score", f"{score} / {TOTAL_EXAMPLES} ({pct}%)")
        with col_best:
            st.metric("Your best score", f"{best} / {TOTAL_EXAMPLES}")

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

    # ---------------------------------------------------------------------------
    # Main quiz loop
    # ---------------------------------------------------------------------------
    else:
        idx = st.session_state.tm_index
        example = TRAINING_EXAMPLES[idx]

        st.caption(f"Example {idx + 1} of {TOTAL_EXAMPLES}  |  "
                   f"Score so far: {st.session_state.tm_score} / {idx}")

        st.progress((idx) / TOTAL_EXAMPLES)

        st.subheader(f"Example {idx + 1}")

        # disabled=True prevents editing — key includes idx so Streamlit creates a new
        # widget for each example rather than reusing the previous one.
        st.text_input("Subject", value=example["subject"], disabled=True,
                      key=f"tm_subj_{idx}")
        st.text_area("Body", value=example["body"], height=160, disabled=True,
                     key=f"tm_body_{idx}")

        st.divider()

        # ---------------------------------------------------------------------------
        # Two-state view: guess form (before submission) / result view (after)
        # ---------------------------------------------------------------------------
        if not st.session_state.tm_submitted:

            st.markdown("**Is this email phishing or legitimate?**")

            # key includes idx so the radio resets when the user moves to a new example.
            guess = st.radio(
                "Your guess",
                ["phishing", "legitimate"],
                horizontal=True,
                key=f"tm_guess_{idx}",
            )

            if st.button("Submit guess", type="primary", key=f"tm_submit_{idx}"):

                try:
                    label, confidence, combined_text = predict_email(
                        model, vec, example["subject"], example["body"]
                    )
                except Exception as exc:
                    st.error(f"Model prediction failed: {exc}")
                    st.stop()

                weights = []
                try:
                    weights = explain_with_lime(
                        explainer, model, vec, combined_text,
                        num_features=LIME_NUM_FEATURES,
                    )
                except Exception:
                    pass  # LIME failure is non-fatal — explanation just won't show

                # Store results in session_state — local variables are lost after st.rerun().
                st.session_state.tm_user_guess = guess
                st.session_state.tm_result_label = label
                st.session_state.tm_result_confidence = confidence
                st.session_state.tm_result_weights = weights

                # Score against ground truth (not the model) — user can be right when
                # the model is wrong.
                if guess == example["label"]:
                    st.session_state.tm_score += 1

                st.session_state.tm_submitted = True
                st.rerun()

        else:
            # --- RESULT VIEW ---

            guess = st.session_state.tm_user_guess
            label = st.session_state.tm_result_label
            confidence = st.session_state.tm_result_confidence
            weights = st.session_state.tm_result_weights
            ground_truth = example["label"]

            user_correct = (guess == ground_truth)

            st.subheader("Result")

            # Three columns: actual label, model prediction, model confidence.
            col_truth, col_model, col_conf = st.columns(3)
            with col_truth:
                if ground_truth == "phishing":
                    st.error(f"Actual: **PHISHING** 🚨")
                else:
                    st.success(f"Actual: **LEGITIMATE** ✅")
            with col_model:
                if label == "phishing":
                    st.error(f"Model: **PHISHING** 🚨")
                else:
                    st.success(f"Model: **LEGITIMATE** ✅")
            with col_conf:
                st.metric("Model confidence", f"{confidence:.1%}")

            if user_correct:
                st.success(f"You guessed **{guess}** — correct!")
            else:
                st.warning(
                    f"You guessed **{guess}** but this was a **{ground_truth}** email."
                )

            # Flag when the model disagrees with the ground truth so the user isn't confused.
            if label != ground_truth:
                st.info(
                    f"Note: the model predicted **{label}** but the correct answer "
                    f"is **{ground_truth}**. The model isn't always right."
                )

            # ---------------------------------------------------------------------------
            # LIME explanation bar chart for the training mode result view
            # ---------------------------------------------------------------------------
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

            # ---------------------------------------------------------------------------
            # Next button — advances to the next example or shows the results screen
            # ---------------------------------------------------------------------------
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
