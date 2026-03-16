"""
preprocess.py — Data preprocessing pipeline for PhishGuard.

This script reads a raw combined dataset CSV from data/raw/dataset.csv,
which is expected to have two columns: 'text_combined' (the concatenated
email subject and body) and 'label' (the class label in various string or
integer formats depending on the source dataset).

It performs the following steps:
  1. Load the raw CSV
  2. Drop rows with missing values in the two key columns
  3. Strip HTML tags and normalise whitespace in the text column
  4. Map all label variants ('spam', 'phishing', 'ham', 'legitimate', etc.)
     to a consistent binary integer: 0 = legitimate, 1 = phishing
  5. Drop any rows whose label could not be mapped
  6. Save the cleaned dataset to data/processed/processed.csv
  7. Print a class balance warning if either class is under 40% of the total

Input  : data/raw/dataset.csv
Output : data/processed/processed.csv
"""

import re
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/dataset.csv")
OUT_PATH = Path("data/processed/processed.csv")


def clean_text(s: str) -> str:
    """
    Clean a single email text string.

    Removes HTML tags (e.g. from email clients that send HTML bodies) and
    collapses any sequence of whitespace characters into a single space.

    Parameters
    ----------
    s : str
        The raw text to clean.

    Returns
    -------
    str
        The cleaned text with HTML stripped and whitespace normalised.
    """
    s = str(s)
    # Remove HTML tags — some emails in the dataset contain raw HTML markup
    s = re.sub(r"<.*?>", " ", s)
    # Collapse runs of whitespace (spaces, tabs, newlines) into one space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    """
    Run the full preprocessing pipeline.

    Reads the raw dataset, cleans the text, normalises the labels, and writes
    the processed output. Also prints dataset statistics and a class balance
    warning if the distribution is skewed beyond the 40/60 threshold.
    """
    # Try to load the raw dataset; give a clear error if the file is missing
    # rather than letting pandas raise a confusing FileNotFoundError
    try:
        df = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        print(
            f"[ERROR] Raw dataset not found at '{RAW_PATH}'. "
            "Please place dataset.csv in the data/raw/ directory before running."
        )
        return

    text_col = "text_combined"
    label_col = "label"

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(clean_text)

    # Normalise label strings to lowercase and strip surrounding whitespace
    # so that the mapping below handles any capitalisation variation
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    # Map every label variant found across the three source datasets to a
    # consistent binary integer (0 = legitimate, 1 = phishing)
    mapping = {
        "0": 0,             # numeric string from datasets that pre-encoded labels
        "1": 1,             # numeric string from datasets that pre-encoded labels
        "legitimate": 0,    # label format used in the Kaggle phishing dataset
        "ham": 0,           # label format used in the Enron corpus
        "not phishing": 0,  # label format used in some PhishTank-derived datasets
        "phishing": 1,      # label format used in the Kaggle phishing dataset
        "spam": 1,          # label format used in the Enron corpus (spam = phishing proxy)
    }
    df[label_col] = df[label_col].map(mapping)

    # Drop rows where the label was not in the mapping (unmappable labels
    # become NaN after .map(), so dropna() removes them cleanly)
    df = df.dropna()
    df[label_col] = df[label_col].astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} rows={len(df)}")
    print(df[label_col].value_counts())

    # --- Class balance check ---
    # A heavily skewed dataset (e.g. 95% phishing, 5% legitimate) can cause
    # the model to appear accurate while actually just predicting the majority
    # class — so we warn early if either class is under 40% of the total
    total = len(df)
    for cls, label_name in [(0, "Legitimate"), (1, "Phishing")]:
        count = (df[label_col] == cls).sum()
        proportion = count / total
        if proportion < 0.40:
            print(
                f"[WARNING] Class imbalance detected: '{label_name}' makes up only "
                f"{proportion:.1%} of the dataset ({count}/{total} samples). "
                "Consider oversampling, undersampling, or using class_weight='balanced'."
            )


if __name__ == "__main__":
    main()
