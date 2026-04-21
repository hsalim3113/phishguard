"""preprocess.py — cleans raw email data and saves a processed CSV.

Reads the raw dataset, applies text-cleaning to every email, standardises
labels to 0/1 integers, and writes a processed CSV that train.py reads.
Run preprocessing separately so you can retrain without re-cleaning each time.

Run from the project root:
    python src/preprocess.py
"""

# re: built-in regular expression library — finds and replaces patterns in text
# (HTML tags, URLs, email addresses, punctuation runs, and whitespace).
import re

# json: saves dataset statistics as a machine-readable file that the Streamlit
# app loads to display the "About this model" panel.
import json

# pandas: reads the CSV, applies the cleaning function to every row with .apply(),
# filters invalid labels with .dropna(), and writes the result back to CSV.
import pandas as pd

# pathlib.Path: builds file paths that work on Windows, macOS, and Linux.
from pathlib import Path

# All file paths come from config.py so changing a location only requires one edit.
from config import (
    DATA_RAW,          # Path to the original, uncleaned dataset CSV
    DATA_PROCESSED,    # Path where the cleaned CSV will be written
    EVAL_DIR,          # Directory for evaluation outputs (stats files go here)
    DATASET_STATS_TXT, # Path for the human-readable stats text file
    DATASET_STATS_JSON, # Path for the machine-readable stats JSON file
)


def clean_text(s: str) -> str:
    """Clean a raw email string so it is ready for TF-IDF vectorisation.

    Applied to every email at both training time (in main()) and prediction time
    (via predict.py). Both paths must use identical cleaning so the model always
    sees the same kind of input.

    Steps in order: strip HTML → replace URLs with [URL] → replace email addresses
    with [EMAIL] → lowercase → remove punctuation (keep .!?) → collapse repeated
    punctuation → collapse whitespace.

    Args:
        s (str): Raw email text (subject + body concatenated).

    Returns:
        str: Cleaned, normalised text ready for the vectoriser.
    """
    # str() guards against NaN values that pandas may pass for empty cells.
    s = str(s)

    # Strip HTML tags — replaced with a space so adjacent words don't merge.
    # WHY a space rather than nothing: "click<b>here</b>" → "click here" not "clickhere".
    s = re.sub(r"<[^>]+>", " ", s)

    # Replace URLs with the token [URL] rather than deleting them.
    # The presence of a URL is itself a phishing signal; the specific domain is not useful
    # because phishing domains change constantly and would just cause overfitting.
    s = re.sub(
        r"https?://\S+|www\.\S+",
        " [URL] ",
        s,
        flags=re.IGNORECASE,
    )

    # Replace email addresses with [EMAIL] for the same reason — their presence
    # is informative but specific addresses overfit to throwaway phishing accounts.
    s = re.sub(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}", " [EMAIL] ", s)

    # Lowercase so "Urgent" and "urgent" are treated as the same word by the vectoriser.
    s = s.lower()

    # Remove symbols that carry no meaning (e.g. *, #, $) but keep .!? because
    # exclamation marks are urgency signals common in phishing emails.
    s = re.sub(r"[^\w\s.!?]", " ", s)

    # Collapse "!!!" → "!" so repeated punctuation doesn't split into multiple rare tokens
    # that each receive too little weight to be useful.
    s = re.sub(r"([.!?]){2,}", r"\1", s)

    # Collapse runs of whitespace to a single space and strip edges.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _save_dataset_stats(df: pd.DataFrame, text_col: str, label_col: str) -> None:
    """Compute dataset statistics and save as .txt and .json files.

    The Streamlit app reads the JSON at startup to show total samples, class
    percentages, and average email lengths in the "About this model" panel —
    without needing access to the full training dataset.

    Args:
        df (pd.DataFrame): Processed DataFrame with cleaned text and integer labels.
        text_col (str): Column name for cleaned email text ("text_combined").
        label_col (str): Column name for integer labels ("label"); 0=legitimate, 1=phishing.

    Returns:
        None. Writes to DATASET_STATS_TXT and DATASET_STATS_JSON.
    """
    # Create the output directory if it doesn't exist yet.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)

    # value_counts() → dict maps label → count; int() converts numpy ints for JSON.
    counts = df[label_col].value_counts().to_dict()
    n_legit = int(counts.get(0, 0))
    n_phish = int(counts.get(1, 0))

    # Compute average text length per class using a temporary column (fast, vectorised),
    # then remove the column so it doesn't appear in the saved CSV.
    df["_text_len"] = df[text_col].str.len()
    avg_len_legit = float(df.loc[df[label_col] == 0, "_text_len"].mean())
    avg_len_phish = float(df.loc[df[label_col] == 1, "_text_len"].mean())
    df.drop(columns=["_text_len"], inplace=True)

    # Count emails that contained at least one URL before cleaning.
    n_urls = int(df[text_col].str.contains(r"\[URL\]", regex=True).sum())

    # All values are plain Python ints/floats so json.dumps() works without a custom encoder.
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

    # First two lines are title and separator; lines[2:] are the data rows printed below.
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

    # Save both formats: .txt for manual inspection, .json for the Streamlit app.
    DATASET_STATS_TXT.write_text("\n".join(lines), encoding="utf-8")
    DATASET_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Saved dataset stats:")
    # Skip title and separator lines when printing
    for line in lines[2:]:
        print(" ", line)


def main() -> None:
    """Load the raw dataset, clean all text, encode labels, and save a processed CSV.

    Steps: read raw CSV → keep only text_combined and label columns → clean text
    → map all label formats to 0/1 integers → drop unrecognised labels → save CSV
    → save dataset statistics.

    Returns:
        None. Writes to DATA_PROCESSED, DATASET_STATS_TXT, and DATASET_STATS_JSON.

    Raises:
        FileNotFoundError: If data/raw/dataset.csv does not exist.
        KeyError: If the CSV lacks "text_combined" or "label" columns.
    """
    # Read the raw CSV — expected columns: "text_combined" and "label".
    df = pd.read_csv(DATA_RAW)

    text_col = "text_combined"
    label_col = "label"

    # Keep only the two columns the model needs; drop rows where either is missing.
    # A missing label has no ground truth; a missing text has nothing to vectorise.
    df = df[[text_col, label_col]].dropna()

    # Apply cleaning to every email. .apply() is faster than a Python for loop
    # because pandas optimises the column iteration internally.
    df[text_col] = df[text_col].apply(clean_text)

    # ---------------------------------------------------------------------------
    # Label normalisation
    # ---------------------------------------------------------------------------
    # Different datasets label emails differently (0/1, phishing/legitimate, spam/ham).
    # This maps all known formats to a consistent integer: 0 = legitimate, 1 = phishing.
    # .str.lower().str.strip() handles mixed case and stray spaces before mapping.
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    mapping = {
        "0": 0, "1": 1,                          # numeric labels already in string form
        "legitimate": 0, "ham": 0, "not phishing": 0,  # text labels for class 0
        "phishing": 1, "spam": 1,                # text labels for class 1
    }
    # .map() replaces each value; anything not in the dict becomes NaN.
    df[label_col] = df[label_col].map(mapping)

    # Drop rows with unrecognised labels (they became NaN after .map()).
    df = df.dropna()

    # Store labels as integers — sklearn classifiers expect integer class indices.
    df[label_col] = df[label_col].astype(int)

    # Create the output directory and save the cleaned DataFrame.
    # index=False omits the pandas row numbers from the CSV.
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"Saved processed data: {DATA_PROCESSED}  rows={len(df)}")

    # Print class counts to confirm the dataset is not severely imbalanced.
    print(df[label_col].value_counts())

    _save_dataset_stats(df, text_col, label_col)


# Only run main() when this script is executed directly, not when imported.
if __name__ == "__main__":
    main()
