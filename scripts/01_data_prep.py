"""
01_data_prep.py
---------------
Load the raw Yelp review dataset, sample 15,000 rows stratified by star rating,
clean the text, and save the processed file to data/yelp_processed.csv.

Expected input:  data/yelp_review.csv  (or yelp_academic_dataset_review.json)
Output:          data/yelp_processed.csv
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_SIZE = 15_000       # total rows to keep
RANDOM_SEED = 42

# ─── 1. Load raw data ─────────────────────────────────────────────────────────

def load_raw(data_dir: str) -> pd.DataFrame:
    """
    Try to load the Yelp dataset from several common filenames/formats.
    Supports both the Kaggle CSV and the official Yelp JSON lines file.
    """
    csv_path  = os.path.join(data_dir, "yelp_review.csv")
    json_path = os.path.join(data_dir, "yelp_academic_dataset_review.json")

    if os.path.exists(csv_path):
        print(f"Loading CSV from {csv_path} …")
        df = pd.read_csv(csv_path, usecols=["text", "stars", "business_id"])
        return df

    if os.path.exists(json_path):
        print(f"Loading JSON lines from {json_path} …")
        records = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading JSON"):
                rec = json.loads(line)
                records.append({
                    "text":        rec.get("text", ""),
                    "stars":       rec.get("stars"),
                    "business_id": rec.get("business_id", ""),
                })
        return pd.DataFrame(records)

    raise FileNotFoundError(
        "No Yelp data file found in data/. "
        "Please place yelp_review.csv or yelp_academic_dataset_review.json in the data/ folder.\n"
        "Download from: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset"
    )


# ─── 2. Clean text ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning: strip HTML entities, non-ASCII, excess whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"[^\x00-\x7F]+", " ", text)    # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── 3. Stratified sample ─────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows with equal representation across star ratings (1–5).
    If a stratum has fewer than n//5 rows, take all of them.
    """
    per_class = n // 5
    parts = []
    for star in [1, 2, 3, 4, 5]:
        subset = df[df["stars"] == star]
        take   = min(len(subset), per_class)
        parts.append(subset.sample(take, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ─── 4. Tokenize for Word2Vec ─────────────────────────────────────────────────

def simple_tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric chars; remove short tokens."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if len(t) > 2]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load
    df = load_raw(DATA_DIR)
    print(f"Raw dataset: {len(df):,} rows")

    # Validate columns
    required = {"text", "stars"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Drop rows with missing text or stars
    df = df.dropna(subset=["text", "stars"])
    df["stars"] = df["stars"].astype(int)
    df = df[df["stars"].between(1, 5)]
    print(f"After dropping nulls/invalid stars: {len(df):,} rows")

    # Sample
    df = stratified_sample(df, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Stratified sample: {len(df):,} rows")
    print(df["stars"].value_counts().sort_index())

    # Clean text
    print("Cleaning text …")
    df["text_clean"] = df["text"].apply(clean_text)

    # Tokenize (stored as space-joined string for easy re-use)
    print("Tokenizing …")
    df["tokens"] = df["text_clean"].apply(
        lambda t: " ".join(simple_tokenize(t))
    )

    # Filter out very short documents (< 10 tokens)
    df = df[df["tokens"].str.split().str.len() >= 10].reset_index(drop=True)
    print(f"After filtering short docs: {len(df):,} rows")

    # Save
    out_path = os.path.join(DATA_DIR, "yelp_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved processed data to {out_path}")
    print(df[["stars", "text_clean", "tokens"]].head(3))


if __name__ == "__main__":
    main()
