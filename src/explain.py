"""
explain.py — Explainability module for PhishGuard.

This module provides two different methods for explaining a prediction,
which is one of the core aims of the project — it's not enough to just
say "phishing", the user should be able to see why.

LIME (Local Interpretable Model-agnostic Explanations):
  LIME works by taking the input email, generating hundreds of slightly
  modified versions of it (randomly masking individual words), running all
  of them through the classifier, and then fitting a simple linear model
  to approximate how the classifier behaves locally around that specific
  input. The output is a set of word-level weights that show which words
  pushed the prediction towards phishing or legitimate for this email.

  Positive weight → word pushed the prediction towards 'phishing'
  Negative weight → word pushed the prediction towards 'legitimate'

  LIME was chosen over alternatives like SHAP because it's well-documented,
  straightforward to apply to text classifiers, and the output is easy to
  present to a non-technical audience.

Coefficient-based explanation (explain_with_coefficients):
  A simpler, faster alternative. Because we're using Logistic Regression,
  the model's learned coefficients directly represent how much each word
  in the vocabulary contributes to a phishing prediction globally. We
  multiply each coefficient by the word's TF-IDF score in the given email
  to get an attribution that's both globally meaningful and scaled to what
  actually appears in the email. It's deterministic, much faster than LIME,
  and gives a complementary global view alongside LIME's local one.
"""

import numpy as np
from lime.lime_text import LimeTextExplainer


def build_explainer():
    """
    Create and return a LimeTextExplainer configured for binary classification.

    The class names need to match the order in the model's predict_proba output:
    index 0 = 'legitimate', index 1 = 'phishing'.

    Returns
    -------
    LimeTextExplainer
        A ready-to-use LIME explainer instance.
    """
    # Without a fixed random seed, LIME's perturbations are different on every
    # call, which means the same email can give slightly different explanations
    # each time — not ideal for a demo or for comparing results across runs
    return LimeTextExplainer(class_names=["legitimate", "phishing"], random_state=42)


def explain_with_lime(explainer, model, vec, text: str, num_features: int = 15):
    """
    Generate a LIME explanation for a single email text.

    We ask LIME for 15 candidate features, then filter down to the 10 with
    the highest absolute weight. This makes sure we always surface the most
    influential words rather than relying on LIME's internal ranking order.

    Parameters
    ----------
    explainer : LimeTextExplainer
        The explainer instance returned by build_explainer().
    model : sklearn estimator
        The trained Logistic Regression classifier.
    vec : TfidfVectorizer
        The fitted TF-IDF vectoriser.
    text : str
        The full email text (subject + body combined).
    num_features : int, optional
        Number of candidate features to request from LIME before we filter
        down to the top 10 by absolute weight (default 15).

    Returns
    -------
    list of tuple
        Up to 10 (word, weight) tuples sorted by absolute weight descending.
        Positive weight → pushes towards phishing.
        Negative weight → pushes towards legitimate.
    """
    def predict_proba(texts):
        # LIME passes its perturbed text variants in here — we transform them
        # exactly the same way we'd transform a real email
        X = vec.transform(texts)
        return model.predict_proba(X)

    # num_samples=1000: LIME generates perturbed versions of the input and fits
    # a local linear model to them. More samples = more stable explanation, but
    # also slower. 1000 is a reasonable middle ground for a live web app —
    # the default of 5000 was noticeably slow during development.
    #
    # distance_metric='cosine': cosine similarity is a better fit for TF-IDF
    # vectors than the default euclidean distance. With sparse high-dimensional
    # vectors, the angle between them (which words appear) matters more than
    # their absolute magnitude, so cosine gives more accurate local weights.
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=num_features,
        num_samples=1000,
        distance_metric="cosine",
    )

    # as_list() returns [(word, weight), ...] — positive means the word pushed
    # toward phishing, negative means it pushed toward legitimate
    all_weights = exp.as_list()

    # LIME doesn't always return results sorted by absolute impact, so we sort
    # here to make sure the table always shows the most influential words first
    top_weights = sorted(all_weights, key=lambda x: abs(x[1]), reverse=True)[:10]
    return top_weights


def explain_with_coefficients(model, vec, text: str, top_n: int = 10):
    """
    Generate a coefficient-based explanation for a single email text.

    Because we're using Logistic Regression, each word in the vocabulary has
    a learned coefficient that represents its global importance to the phishing
    prediction. We scale each coefficient by the word's TF-IDF score in the
    given email so the output reflects both global importance and local presence.

    This is faster and deterministic compared to LIME, and gives a useful
    second perspective — LIME tells you which words mattered in this specific
    email, while this tells you which words the model learned to associate with
    phishing across the entire training dataset.

    Parameters
    ----------
    model : sklearn LogisticRegression
        The trained classifier (must have a .coef_ attribute).
    vec : TfidfVectorizer
        The fitted vectoriser (must have .vocabulary_ and .get_feature_names_out()).
    text : str
        The full email text (subject + body combined).
    top_n : int, optional
        How many words to return, sorted by absolute score (default 10).

    Returns
    -------
    list of tuple
        Up to top_n (word, score) tuples sorted by absolute score descending.
        Positive score = word is a phishing indicator.
        Negative score = word is associated with legitimate emails.
    """
    # Transform the email into a TF-IDF vector — same process as during training,
    # just for a single document instead of the full training set
    X = vec.transform([text])

    # coef_[0] holds the model's learned weight for every word in the vocabulary.
    # A high positive coefficient means that word strongly indicates phishing
    # globally. We multiply by the TF-IDF score to scale by how much the word
    # actually appears in this email — a globally important word that doesn't
    # appear here shouldn't dominate the local explanation
    coefficients = model.coef_[0]
    tfidf_scores = np.asarray(X.todense()).flatten()

    # attribution[i] = how much word i contributed to the prediction for this email
    attribution = coefficients * tfidf_scores

    # Only include words that actually appear in this email —
    # words with a TF-IDF score of zero have zero contribution here
    feature_names = vec.get_feature_names_out()
    word_scores = [
        (feature_names[i], float(attribution[i]))
        for i in range(len(attribution))
        if tfidf_scores[i] > 0
    ]

    # Sort by absolute value so the most influential words (positive or negative)
    # rise to the top regardless of which direction they push the prediction
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    return word_scores[:top_n]
