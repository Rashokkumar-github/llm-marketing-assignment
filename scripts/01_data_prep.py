"""
01_data_prep.py
---------------
Download the Yelp restaurant reviews dataset via kagglehub, sample 15,000 rows
stratified by star rating, clean the text, and save to data/yelp_processed.csv.

Dataset:  farukalam/yelp-restaurant-reviews  (Kaggle)
Output:   data/yelp_processed.csv
"""

import os
import re
import glob
import pandas as pd
import kagglehub
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_SIZE = 15_000
RANDOM_SEED = 42

# Column name variants to try for review text and star rating
TEXT_ALIASES   = ["text", "review", "Review", "review_text", "Review_Text", "body"]
RATING_ALIASES = ["stars", "rating", "Rating", "star_rating", "Stars", "score"]

# ─── 1. Download and load ──────────────────────────────────────────────────────

def download_dataset() -> str:
    """Download dataset via kagglehub and return path to the downloaded folder."""
    print("Downloading farukalam/yelp-restaurant-reviews via kagglehub …")
    path = kagglehub.dataset_download("farukalam/yelp-restaurant-reviews")
    print(f"Downloaded to: {path}")
    return path


def load_raw(download_path: str) -> pd.DataFrame:
    """
    Find the CSV file in the downloaded folder, load it, and normalise
    column names to 'text' and 'stars' regardless of the original naming.
    """
    csv_files = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset at {download_path}")

    csv_path = csv_files[0]
    print(f"Loading CSV: {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"Columns found: {list(df.columns)}")

    # Normalise text column
    text_col = next((c for c in TEXT_ALIASES if c in df.columns), None)
    if text_col is None:
        raise ValueError(
            f"Could not find a review text column. Available columns: {list(df.columns)}\n"
            f"Expected one of: {TEXT_ALIASES}"
        )

    # Normalise rating column
    rating_col = next((c for c in RATING_ALIASES if c in df.columns), None)
    if rating_col is None:
        raise ValueError(
            f"Could not find a star rating column. Available columns: {list(df.columns)}\n"
            f"Expected one of: {RATING_ALIASES}"
        )

    df = df.rename(columns={text_col: "text", rating_col: "stars"})

    # Keep business_id if it exists (used for subgroup analysis)
    if "business_id" not in df.columns:
        df["business_id"] = "unknown"

    return df[["text", "stars", "business_id"] + [
        c for c in df.columns if c not in ("text", "stars", "business_id")
    ]]


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

    # Download and load
    download_path = download_dataset()
    df = load_raw(download_path)
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
