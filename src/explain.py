"""explain.py — LIME and SHAP explanation helpers for the phishing detector.

Returns (word, weight) lists so the Streamlit app can render colour-coded bar charts.
Positive weights push toward phishing; negative weights push toward legitimate.
"""

# LimeTextExplainer: generates explanations by masking words and measuring
# how each word's removal changes the phishing probability.
from lime.lime_text import LimeTextExplainer

# Feature count constants so bar chart sizes are configurable from config.py.
from config import LIME_NUM_FEATURES, SHAP_NUM_FEATURES


def build_explainer() -> LimeTextExplainer:
    """Construct a LIME text explainer configured for binary phishing classification.

    Returns:
        LimeTextExplainer: Configured explainer; call explain_instance() to generate explanations.
    """
    # class_names must match the label encoding: index 0 = legitimate, index 1 = phishing.
    return LimeTextExplainer(class_names=["legitimate", "phishing"])


def explain_with_lime(
    explainer: LimeTextExplainer,
    model,
    vec,
    text: str,
    num_features: int = LIME_NUM_FEATURES,
) -> list[tuple[str, float]]:
    """Generate a LIME explanation showing which words most influenced the prediction.

    LIME masks random word combinations (~5,000 perturbations), scores each with the
    model, and fits a local linear model whose coefficients become the word weights.

    Args:
        explainer (LimeTextExplainer): Built by build_explainer().
        model: Fitted sklearn classifier with predict_proba().
        vec: Fitted TF-IDF vectoriser — must be the same one used during training.
        text (str): The combined_text returned by predict_email().
        num_features (int): Number of top words to return. Default: LIME_NUM_FEATURES.

    Returns:
        list[tuple[str, float]]: (word, weight) pairs sorted by descending absolute weight.
    """
    # Closure so LIME can call vec.transform + model.predict_proba on its perturbations.
    def _predict_proba(texts):
        return model.predict_proba(vec.transform(texts))

    # explain_instance tokenises text, creates perturbations, scores them, and fits
    # a local linear model — num_features controls how many words are returned.
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=_predict_proba,
        num_features=num_features,
    )

    # as_list() returns (word, weight) pairs for the phishing class, sorted by abs weight.
    return exp.as_list()


def explain_with_shap(
    model,
    vec,
    text: str,
    num_features: int = SHAP_NUM_FEATURES,
) -> list[tuple[str, float]]:
    """Generate a SHAP explanation for a Logistic Regression model using LinearExplainer.

    LinearExplainer reads the model's coefficient array directly — values are
    mathematically exact (no sampling), unlike LIME.

    Args:
        model: Fitted LogisticRegression classifier.
        vec: Fitted TF-IDF vectoriser used during training.
        text (str): The email text to explain (combined subject + body, post-truncation).
        num_features (int): How many top words to return. Default: SHAP_NUM_FEATURES.

    Returns:
        list[tuple[str, float]]: Top (word, SHAP-value) pairs sorted by descending absolute value.
            Returns an empty list if SHAP fails for any reason.
    """
    try:
        import shap
        import numpy as np

        # Transform to TF-IDF sparse matrix — shape (1, TFIDF_MAX_FEATURES).
        X = vec.transform([text])

        feature_names = vec.get_feature_names_out()

        # Zero vector = baseline prediction with no words present.
        # SHAP values measure each word's contribution relative to this baseline.
        background = np.zeros((1, X.shape[1]))

        # LinearExplainer reads model coefficients directly — only valid for linear models.
        explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")

        shap_values = explainer.shap_values(X)

        # Handle both old (list per class) and new (single array) shap output formats.
        # We always want phishing class (index 1 in old format).
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        # Sort by absolute value descending and take top num_features.
        indices = np.argsort(np.abs(vals))[::-1][:num_features]

        return [(feature_names[i], float(vals[i])) for i in indices]

    except Exception as exc:  # noqa: BLE001
        # Return empty list on any failure so the app shows a warning instead of crashing.
        print(f"[explain_with_shap] failed: {exc}")
        return []


def explain_with_shap_xgb(
    model,
    vec,
    text: str,
    num_features: int = SHAP_NUM_FEATURES,
) -> list[tuple[str, float]]:
    """Generate a SHAP explanation for an XGBoost model using TreeExplainer.

    TreeExplainer traverses the actual decision trees to compute exact Shapley values —
    correct for tree-based models where LinearExplainer would be invalid.

    Args:
        model: Fitted XGBClassifier saved to MODEL_XGB by train.py.
        vec: Fitted TF-IDF vectoriser (must be the same one used at training time).
        text (str): The email text to explain.
        num_features (int): How many top words to return. Default: SHAP_NUM_FEATURES.

    Returns:
        list[tuple[str, float]]: Top (word, SHAP-value) pairs sorted by descending absolute value.
            Returns an empty list if SHAP fails for any reason.
    """
    try:
        import shap
        import numpy as np

        # Sparse matrix — XGBoost handles scipy sparse natively, no need to densify.
        vector = vec.transform([text])

        # TreeExplainer analyses the XGBoost tree structure directly — no background needed.
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(vector)

        feature_names = vec.get_feature_names_out()

        # Handle both old (list per class) and new (single array) shap output formats.
        if isinstance(shap_values, list):
            vals = shap_values[1].flatten()
        else:
            vals = shap_values.flatten()

        # Sort by absolute value descending and take top num_features.
        indices = np.argsort(np.abs(vals))[::-1][:num_features]

        return [(feature_names[i], float(vals[i])) for i in indices]

    except Exception as exc:  # noqa: BLE001
        print(f"[explain_with_shap_xgb] failed: {exc}")
        return []
