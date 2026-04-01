"""explain.py — LIME and SHAP explanation helpers for the phishing detector.

Both functions return a list of (feature_name, weight) tuples that the
Streamlit app renders as colour-coded bar charts.
"""

from lime.lime_text import LimeTextExplainer

from config import LIME_NUM_FEATURES, SHAP_NUM_FEATURES


def build_explainer() -> LimeTextExplainer:
    """Construct a LIME text explainer with the correct class ordering.

    Returns:
        LimeTextExplainer: Explainer configured for binary classification
            where index 0 = legitimate and index 1 = phishing.
    """
    return LimeTextExplainer(class_names=["legitimate", "phishing"])


def explain_with_lime(
    explainer: LimeTextExplainer,
    model,
    vec,
    text: str,
    num_features: int = LIME_NUM_FEATURES,
) -> list[tuple[str, float]]:
    """Generate a LIME explanation for a single email text.

    Positive weights push the prediction toward *phishing*; negative weights
    push toward *legitimate*.

    Args:
        explainer (LimeTextExplainer): Built by :func:`build_explainer`.
        model: Fitted sklearn classifier with ``predict_proba``.
        vec: Fitted TF-IDF vectoriser.
        text (str): Combined (subject + body) email text.
        num_features (int): Number of top features to return.

    Returns:
        list[tuple[str, float]]: Feature–weight pairs sorted by descending
            absolute weight.

    Raises:
        ValueError: If ``text`` is empty or contains fewer than two tokens.
    """
    # LIME needs a function that takes raw strings, so wrap the vectoriser and
    # model together into one callable it can call repeatedly with perturbed text.
    def _predict_proba(texts):
        return model.predict_proba(vec.transform(texts))

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=_predict_proba,
        num_features=num_features,
    )
    # as_list returns [(word, weight), ...] already sorted by descending weight
    return exp.as_list()


def explain_with_shap(
    model,
    vec,
    text: str,
    num_features: int = SHAP_NUM_FEATURES,
) -> list[tuple[str, float]]:
    """Generate a SHAP explanation for a single email text using LinearExplainer.

    Only compatible with linear models (e.g. LogisticRegression).  Positive
    SHAP values indicate contribution toward the *phishing* class.

    Args:
        model: Fitted sklearn linear classifier (e.g. LogisticRegression).
        vec: Fitted TF-IDF vectoriser.
        text (str): Combined email text to explain.
        num_features (int): Number of top features to return.

    Returns:
        list[tuple[str, float]]: Top feature–SHAP-value pairs, sorted by
            descending absolute SHAP value.  Returns an empty list on failure.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    try:
        import shap
        import numpy as np

        X = vec.transform([text])
        feature_names = vec.get_feature_names_out()

        # LinearExplainer reads the model's coefficients directly, which is why
        # it only works with linear models like Logistic Regression. Zeros make
        # sense as the background because most TF-IDF values are 0 for any email.
        background = np.zeros((1, X.shape[1]))
        explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")

        shap_values = explainer.shap_values(X)

        # Positive values mean the word pushed toward phishing, negative means legitimate.
        # Older shap versions return a list per class, newer ones return a single array,
        # so check for both.
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        # Sort by absolute value so the biggest contributors appear first,
        # regardless of whether they push toward phishing or legitimate.
        indices = np.argsort(np.abs(vals))[::-1][:num_features]
        return [(feature_names[i], float(vals[i])) for i in indices]

    except Exception as exc:  # noqa: BLE001
        # If SHAP fails for any reason, return empty so the app doesn't crash
        print(f"[explain_with_shap] failed: {exc}")
        return []
