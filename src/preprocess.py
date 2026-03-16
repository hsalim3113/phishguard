"""
preprocess.py — Data preprocessing pipeline for PhishGuard.

This script takes the raw combined dataset and gets it into a clean,
consistent format that the training script can use directly.

The raw CSV (data/raw/dataset.csv) is expected to have two columns:
  - 'text_combined' : the email subject and body joined into one string
  - 'label'         : the class label, which varies by source dataset
                      (e.g. 'spam', 'ham', 'phishing', '0', '1', etc.)

What this script does:
  1. Load the raw CSV (with a proper error message if it's missing)
  2. Drop rows with missing values in either key column
  3. Strip HTML tags and collapse whitespace in the text column
  4. Map all the different label formats to a consistent 0/1 integer
  5. Drop any rows whose label couldn't be mapped
  6. Save the cleaned dataset to data/processed/processed.csv
  7. Print a warning if either class is under 40% of the total

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

    Some emails in the dataset were scraped with their HTML intact, so we
    strip the tags out before training. We also collapse whitespace so the
    vectoriser doesn't treat double-spaced text differently from single-spaced.

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
    # Some emails come through with full HTML markup still in the body —
    # stripping it out so the model doesn't accidentally learn HTML tag patterns
    s = re.sub(r"<.*?>", " ", s)
    # Collapse any run of whitespace (tabs, newlines, multiple spaces) into
    # a single space so the TF-IDF vectoriser sees consistent token boundaries
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    """
    Run the full preprocessing pipeline.

    Reads the raw dataset, cleans the text, normalises the labels, saves the
    output, and prints a class balance warning if the distribution looks skewed.
    """
    # Catch a missing file before pandas does — the default FileNotFoundError
    # message is easy to miss when running from a different working directory
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

    # Drop rows that are missing either the text or the label —
    # we can't do anything useful with incomplete samples
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(clean_text)

    # Lowercase and strip before mapping so "Spam", "SPAM", and "spam" all
    # resolve to the same key — the source datasets aren't consistent here
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    # The three datasets use completely different label formats, so we unify
    # everything to a binary integer: 0 = legitimate, 1 = phishing
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

    # .map() silently turns any unrecognised label into NaN, so dropna()
    # takes care of cleaning up any rows we couldn't handle
    df = df.dropna()
    df[label_col] = df[label_col].astype(int)

    # Create the output directory if it doesn't exist yet
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} rows={len(df)}")
    print(df[label_col].value_counts())

    # --- Class balance check ---
    # Class imbalance is a known problem with phishing datasets — if one class
    # dominates, the model can look accurate just by always predicting that class.
    # 40% is a rough threshold; anything below it is worth looking into before
    # training, whether that means oversampling, undersampling, or adjusting
    # class weights in the classifier
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
