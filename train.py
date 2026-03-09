import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_PATH = Path("data/processed/processed.csv")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "model_logreg.joblib"
VEC_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    X = df["text_combined"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, digits=3))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    dump(vec, VEC_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VEC_PATH}")

if __name__ == "__main__":
    main()
