"""preprocess.py — cleans raw email data and saves a processed CSV.

Run from the project root:
    python src/preprocess.py
"""

import re
import json
import pandas as pd
from pathlib import Path

from config import (
    DATA_RAW,
    DATA_PROCESSED,
    EVAL_DIR,
    DATASET_STATS_TXT,
    DATASET_STATS_JSON,
)


def clean_text(s: str) -> str:
    """Clean a raw email text string.

    Steps applied in order:
    1. Strip HTML tags.
    2. Replace URLs with the token ``[URL]``.
    3. Replace email addresses with ``[EMAIL]``.
    4. Convert to lowercase.
    5. Remove excessive punctuation (keeps sentence-ending `.!?`).
    6. Collapse whitespace and strip leading/trailing spaces.

    Args:
        s (str): Raw text to clean.

    Returns:
        str: Cleaned text.
    """
    s = str(s)
    # Strip HTML tags, emails often have markup in them that isn't actual text
    s = re.sub(r"<[^>]+>", " ", s)

    # Replace URLs and email addresses with tokens instead of deleting them.
    # Whether an email contains a URL or address is still useful information
    # for the model, even if the specific value changes every time.
    s = re.sub(
        r"https?://\S+|www\.\S+",
        " [URL] ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}", " [EMAIL] ", s)

    # Lowercase and strip punctuation that doesn't add meaning
    s = s.lower()
    s = re.sub(r"[^\w\s.!?]", " ", s)
    s = re.sub(r"([.!?]){2,}", r"\1", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _save_dataset_stats(df: pd.DataFrame, text_col: str, label_col: str) -> None:
    """Compute and persist dataset statistics used by the Streamlit app.

    Args:
        df (pd.DataFrame): Processed dataframe with text and label columns.
        text_col (str): Name of the text column.
        label_col (str): Name of the label column (0 = legitimate, 1 = phishing).

    Returns:
        None
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    counts = df[label_col].value_counts().to_dict()
    n_legit = int(counts.get(0, 0))
    n_phish = int(counts.get(1, 0))

    # Compute average email length per class using a temporary column, then remove it
    df["_text_len"] = df[text_col].str.len()
    avg_len_legit = float(df.loc[df[label_col] == 0, "_text_len"].mean())
    avg_len_phish = float(df.loc[df[label_col] == 1, "_text_len"].mean())
    df.drop(columns=["_text_len"], inplace=True)

    n_urls = int(df[text_col].str.contains(r"\[URL\]", regex=True).sum())

    stats = {
        "total_samples": total,
        "n_legitimate": n_legit,
        "n_phishing": n_phish,
        "pct_legitimate": round(n_legit / total * 100, 2) if total else 0,
        "pct_phishing": round(n_phish / total * 100, 2) if total else 0,
        "avg_text_len_legitimate": round(avg_len_legit, 1),
        "avg_text_len_phishing": round(avg_len_phish, 1),
        "n_samples_with_url": n_urls,
    }

    # Save as plain text for quick reading and JSON for the app to load
    lines = [
        "Dataset Statistics",
        "==================",
        f"Total samples          : {total}",
        f"Legitimate (label=0)   : {n_legit} ({stats['pct_legitimate']}%)",
        f"Phishing   (label=1)   : {n_phish} ({stats['pct_phishing']}%)",
        f"Avg text len — legit   : {stats['avg_text_len_legitimate']} chars",
        f"Avg text len — phishing: {stats['avg_text_len_phishing']} chars",
        f"Samples containing URL : {n_urls}",
    ]
    DATASET_STATS_TXT.write_text("\n".join(lines), encoding="utf-8")
    DATASET_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Saved dataset stats:")
    for line in lines[2:]:
        print(" ", line)


def main() -> None:
    """Load raw dataset, clean text, encode labels, and save processed CSV.

    Returns:
        None

    Raises:
        FileNotFoundError: If the raw dataset CSV does not exist.
        KeyError: If expected columns are missing from the dataset.
    """
    df = pd.read_csv(DATA_RAW)

    text_col = "text_combined"
    label_col = "label"

    # Keep only the two columns we need and drop any rows where either is missing
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(clean_text)

    # Different datasets label emails differently ("phishing", "1", "spam" etc.)
    # so map everything to the same format: 0 = legitimate, 1 = phishing.
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    mapping = {
        "0": 0, "1": 1,
        "legitimate": 0, "ham": 0, "not phishing": 0,
        "phishing": 1, "spam": 1,
    }
    df[label_col] = df[label_col].map(mapping)
    # Any label not in the mapping becomes NaN — drop those rows rather than guessing
    df = df.dropna()
    # Store the label as an integer so sklearn doesn't treat it as a float
    df[label_col] = df[label_col].astype(int)

    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"Saved processed data: {DATA_PROCESSED}  rows={len(df)}")
    print(df[label_col].value_counts())

    _save_dataset_stats(df, text_col, label_col)


if __name__ == "__main__":
    main()
