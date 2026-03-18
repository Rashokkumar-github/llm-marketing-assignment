"""
01_data_prep.py
---------------
Load the Yelp Restaurant Reviews dataset, sample 15,000 rows stratified by
star rating, clean the text, and save to data/yelp_processed.csv.

Input:   data/Yelp Restaurant Reviews.csv
Output:  data/yelp_processed.csv
"""

import os
import re
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_CSV     = os.path.join(DATA_DIR, "Yelp Restaurant Reviews.csv")
OUT_CSV     = os.path.join(DATA_DIR, "yelp_processed.csv")

SAMPLE_SIZE = 15_000
RANDOM_SEED = 42


# ─── 1. Load ──────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Load the raw CSV and normalise to standard column names."""
    print(f"Loading {RAW_CSV} …")
    df = pd.read_csv(RAW_CSV)

    # Rename to standard names used throughout the pipeline
    df = df.rename(columns={
        "Review Text": "text",
        "Rating":      "stars",
        "Yelp URL":    "url",
        "Date":        "date",
    })

    # Extract a business slug from the URL to use as business_id
    # e.g. "https://www.yelp.com/biz/sidney-dairy-barn-sidney" → "sidney-dairy-barn-sidney"
    df["business_id"] = df["url"].str.extract(r"/biz/([^/?]+)")

    return df[["text", "stars", "business_id", "date"]]


# ─── 2. Clean text ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip HTML tags, non-ASCII characters, and excess whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── 3. Stratified sample ─────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows with equal representation across star ratings (1–5).
    Takes min(per_class, available) rows per stratum.
    """
    per_class = n // 5
    parts = []
    for star in [1, 2, 3, 4, 5]:
        subset = df[df["stars"] == star]
        take   = min(len(subset), per_class)
        parts.append(subset.sample(take, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ─── 4. Tokenise for Word2Vec ─────────────────────────────────────────────────

def simple_tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, drop tokens shorter than 3 chars."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if len(t) > 2]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_raw()
    print(f"Raw dataset: {len(df):,} rows")
    print(f"Rating distribution:\n{df['stars'].value_counts().sort_index()}\n")

    # Cast and validate ratings
    df["stars"] = df["stars"].astype(int)
    df = df[df["stars"].between(1, 5)].dropna(subset=["text"])

    # Stratified sample
    df = stratified_sample(df, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Sampled {len(df):,} rows (stratified by star rating)")
    print(df["stars"].value_counts().sort_index())

    # Clean text
    print("\nCleaning text …")
    df["text_clean"] = df["text"].apply(clean_text)

    # Tokenise (space-joined string for easy re-use in Word2Vec)
    print("Tokenising …")
    df["tokens"] = df["text_clean"].apply(lambda t: " ".join(simple_tokenize(t)))

    # Drop very short documents (< 10 tokens)
    before = len(df)
    df = df[df["tokens"].str.split().str.len() >= 10].reset_index(drop=True)
    print(f"Dropped {before - len(df)} docs with < 10 tokens → {len(df):,} remaining")

    # Save
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
    print(df[["stars", "business_id", "text_clean"]].head(3).to_string())


if __name__ == "__main__":
    main()
