from lime.lime_text import LimeTextExplainer

def build_explainer():
    return LimeTextExplainer(class_names=["legitimate", "phishing"])

def explain_with_lime(explainer, model, vec, text: str, num_features: int = 10):
    def predict_proba(texts):
        X = vec.transform(texts)
        return model.predict_proba(X)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=num_features
    )
    return exp.as_list()
