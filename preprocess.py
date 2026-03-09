import re
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/dataset.csv")
OUT_PATH = Path("data/processed/processed.csv")

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    df = pd.read_csv(RAW_PATH)

    text_col = "text_combined"
    label_col = "label"

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(clean_text)

    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    mapping = {
        "0": 0, "1": 1,
        "legitimate": 0, "ham": 0, "not phishing": 0,
        "phishing": 1, "spam": 1
    }
    df[label_col] = df[label_col].map(mapping)

    df = df.dropna()
    df[label_col] = df[label_col].astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} rows={len(df)}")
    print(df[label_col].value_counts())

if __name__ == "__main__":
    main()
