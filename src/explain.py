"""
explain.py — Explainability module for PhishGuard.

This module provides two methods for explaining why the model made a
particular prediction, which is important for user trust and transparency.

LIME (Local Interpretable Model-agnostic Explanations):
  LIME was chosen because it is model-agnostic — it works with any classifier
  by perturbing the input text (randomly masking words) and fitting a simple
  linear model locally around that instance. This means we get a human-readable
  explanation in terms of individual words or phrases rather than just a
  confidence score. For a phishing detector, being able to tell the user
  *why* an email looks suspicious is just as important as the prediction itself.

  The output of explain_with_lime is a list of (word, weight) tuples where:
    - Positive weight → word pushes the prediction towards 'phishing'
    - Negative weight → word pushes the prediction towards 'legitimate'

Coefficient-based explanation (explain_with_coefficients):
  A simpler, faster alternative that multiplies the model's learned coefficients
  by the TF-IDF score of each word in the input. This gives a direct measure of
  how much each word contributed to the prediction based on what the model
  learned during training. It is less locally faithful than LIME but is
  deterministic and much quicker to compute.
"""

import numpy as np
from lime.lime_text import LimeTextExplainer


def build_explainer():
    """
    Create and return a LimeTextExplainer configured for binary classification.

    The class names must match the order used by the model's predict_proba
    output: index 0 = 'legitimate', index 1 = 'phishing'.

    Returns
    -------
    LimeTextExplainer
        A configured LIME explainer instance ready to explain text predictions.
    """
    return LimeTextExplainer(class_names=["legitimate", "phishing"])


def explain_with_lime(explainer, model, vec, text: str, num_features: int = 10):
    """
    Generate a LIME explanation for a single email text.

    LIME works by creating slightly modified versions of the input (by hiding
    individual words) and observing how the model's prediction changes. It then
    fits a simple linear model to approximate the classifier's behaviour
    locally around this specific input.

    Parameters
    ----------
    explainer : LimeTextExplainer
        The LIME explainer instance returned by build_explainer().
    model : sklearn estimator
        The trained Logistic Regression model with a predict_proba method.
    vec : TfidfVectorizer
        The fitted TF-IDF vectoriser used to transform text for the model.
    text : str
        The combined email text (subject + body) to explain.
    num_features : int, optional
        The number of top contributing words/phrases to return (default 10).

    Returns
    -------
    list of tuple
        A list of (word, weight) tuples sorted by absolute weight descending.
        Positive weight → pushes towards phishing.
        Negative weight → pushes towards legitimate.
    """
    def predict_proba(texts):
        # Transform each perturbed text the same way we transform real inputs
        X = vec.transform(texts)
        return model.predict_proba(X)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=num_features
    )

    # exp.as_list() returns [(word, weight), ...] where:
    #   positive weight → that word pushes the model towards predicting phishing
    #   negative weight → that word pushes the model towards predicting legitimate
    return exp.as_list()


def explain_with_coefficients(model, vec, text: str, top_n: int = 10):
    """
    Generate a simpler coefficient-based explanation for a single email text.

    This multiplies each word's learned model coefficient by its TF-IDF score
    in the given text, giving a direct attribution score for every word that
    actually appears in the email. It is faster and deterministic compared to
    LIME, making it a useful fallback or cross-check.

    Positive scores indicate words that contributed towards a phishing
    prediction; negative scores indicate words that contributed towards a
    legitimate prediction.

    Parameters
    ----------
    model : sklearn LogisticRegression
        The trained Logistic Regression model (must have .coef_ attribute).
    vec : TfidfVectorizer
        The fitted TF-IDF vectoriser (must have .vocabulary_ attribute).
    text : str
        The combined email text (subject + body) to explain.
    top_n : int, optional
        The number of top words by absolute score to return (default 10).

    Returns
    -------
    list of tuple
        A list of (word, score) tuples for the top_n words sorted by absolute
        score descending. Positive score = pushes towards phishing,
        negative score = pushes towards legitimate.
    """
    # Transform the input text to a TF-IDF vector (1 x vocab_size sparse matrix)
    X = vec.transform([text])

    # model.coef_[0] gives the coefficient for each feature for class 1 (phishing)
    # Multiplying by the TF-IDF value weights by how prominent the word is in
    # this specific email, not just how discriminative it is globally
    coefficients = model.coef_[0]
    tfidf_scores = np.asarray(X.todense()).flatten()

    # Element-wise product: attribution = coefficient * TF-IDF weight
    attribution = coefficients * tfidf_scores

    # Build a list of (word, score) for words that actually appear in the email
    # (i.e. where the TF-IDF score is non-zero)
    feature_names = vec.get_feature_names_out()
    word_scores = [
        (feature_names[i], float(attribution[i]))
        for i in range(len(attribution))
        if tfidf_scores[i] > 0  # only include words present in this email
    ]

    # Sort by absolute score so the most influential words appear first
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    return word_scores[:top_n]
