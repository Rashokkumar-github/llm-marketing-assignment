"""
03_topic_sentiment.py
---------------------
Run BERTopic on the Yelp corpus (document-level) and compute VADER sentiment scores.
Produces:
  - Topic assignments and probabilities saved to data/yelp_topics.csv
  - Aggregated topic-level sentiment
  - Subgroup comparison: High-rating (4–5★) vs Low-rating (1–2★)
  - Figures: topic prevalence bar chart, topic sentiment chart, subgroup comparison

Input:   data/yelp_processed.csv
Output:  data/yelp_topics.csv
         outputs/fig_topic_prevalence.png
         outputs/fig_topic_sentiment.png
         outputs/fig_subgroup_comparison.png
         outputs/topic_labels.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "yelp_processed.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
TOPICS_OUT  = os.path.join(BASE_DIR, "data", "yelp_topics.csv")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, high-quality
NR_TOPICS       = "auto"                # let BERTopic decide
TOP_N_WORDS     = 10                    # words per topic to display
SEED            = 42


# ─── 1. BERTopic ──────────────────────────────────────────────────────────────

def run_bertopic(docs: list[str]) -> tuple[BERTopic, list[int], np.ndarray]:
    """
    Fit BERTopic on document-level texts using sentence-transformers embeddings.
    Returns (model, topic_ids, probabilities).

    Design choice: document-level topics because Yelp reviews typically describe
    one overall dining/service experience — a single dominant topic per review.
    """
    print(f"Encoding {len(docs):,} documents with '{EMBEDDING_MODEL}' …")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64)

    print("Fitting BERTopic …")
    topic_model = BERTopic(
        embedding_model=embedder,
        nr_topics=NR_TOPICS,
        calculate_probabilities=True,
        verbose=True,
        seed_topic_list=None,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs


def print_topic_summary(topic_model: BERTopic):
    """Print the top words for each discovered topic."""
    topic_info = topic_model.get_topic_info()
    print(f"\n--- Topic Summary ({len(topic_info) - 1} topics, excluding outlier -1) ---")
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        top_words = [w for w, _ in words[:TOP_N_WORDS]]
        print(f"  Topic {tid:3d} | Count: {row['Count']:5d} | Words: {', '.join(top_words)}")


def save_topic_labels(topic_model: BERTopic, out_path: str):
    """
    Save topic top-words to a text file for manual label assignment.
    The researcher fills in the 'Label' column after inspecting exemplar documents.
    """
    lines = ["Topic ID | Top Words | Suggested Label (fill in manually)\n", "-" * 70 + "\n"]
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        top_words = [w for w, _ in words[:8]]
        lines.append(f"Topic {tid:3d} | {', '.join(top_words)} | ??? \n")
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"Topic label template saved → {out_path}")


# ─── 2. VADER sentiment ───────────────────────────────────────────────────────

def compute_vader_sentiment(texts: list[str]) -> list[float]:
    """Return VADER compound scores (−1 = very negative, +1 = very positive)."""
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    return scores


# ─── 3. Aggregate to topic level ─────────────────────────────────────────────

def topic_level_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate document-level VADER scores to the topic level.
    Returns a DataFrame with mean sentiment per topic (excluding outlier -1).
    """
    agg = (
        df[df["topic"] != -1]
        .groupby("topic")
        .agg(
            n_docs        = ("topic",     "count"),
            mean_sentiment= ("sentiment", "mean"),
            std_sentiment = ("sentiment", "std"),
        )
        .reset_index()
        .sort_values("mean_sentiment", ascending=False)
    )
    return agg


# ─── 4. Visualisations ────────────────────────────────────────────────────────

def plot_topic_prevalence(df: pd.DataFrame, topic_model: BERTopic):
    """Bar chart of document count per topic (top 20)."""
    counts = (
        df[df["topic"] != -1]["topic"]
        .value_counts()
        .head(20)
        .reset_index()
    )
    counts.columns = ["topic", "count"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=counts, x="count", y=counts["topic"].astype(str), ax=ax, palette="Blues_d")
    ax.set_xlabel("Number of Documents")
    ax.set_ylabel("Topic ID")
    ax.set_title("Topic Prevalence (Top 20 Topics)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_topic_prevalence.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved → {out}")


def plot_topic_sentiment(agg: pd.DataFrame):
    """Bar chart of mean VADER compound score per topic."""
    top_agg = agg.head(20)
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in top_agg["mean_sentiment"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_agg["topic"].astype(str), top_agg["mean_sentiment"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean VADER Compound Score")
    ax.set_ylabel("Topic ID")
    ax.set_title("Topic-Level Sentiment (Top 20 Topics)")
    ax.invert_yaxis()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_topic_sentiment.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved → {out}")


def plot_subgroup_comparison(df: pd.DataFrame, top_n_topics: int = 10):
    """
    Compare topic prevalence and mean sentiment between
    High-rating (4–5★) and Low-rating (1–2★) subgroups.
    """
    df_sub = df[df["topic"] != -1].copy()

    # Define subgroups
    df_sub["group"] = pd.cut(
        df_sub["stars"],
        bins=[0, 2, 3, 5],
        labels=["Low (1-2★)", "Mid (3★)", "High (4-5★)"]
    )

    # Focus on High vs Low
    df_hl = df_sub[df_sub["group"].isin(["Low (1-2★)", "High (4-5★)"])].copy()

    # Top topics by overall count
    top_topics = df_sub["topic"].value_counts().head(top_n_topics).index.tolist()
    df_hl = df_hl[df_hl["topic"].isin(top_topics)]

    # ── Topic prevalence ──
    total_per_group = df_hl.groupby("group").size()
    prev = (
        df_hl.groupby(["group", "topic"])
        .size()
        .reset_index(name="count")
    )
    prev["share"] = prev.apply(
        lambda r: r["count"] / total_per_group[r["group"]], axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: prevalence
    prev_pivot = prev.pivot(index="topic", columns="group", values="share").fillna(0)
    prev_pivot.plot(kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"], alpha=0.85)
    axes[0].set_title("Topic Prevalence: High vs Low Ratings")
    axes[0].set_xlabel("Topic ID")
    axes[0].set_ylabel("Share of Documents")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Rating Group")

    # Panel 2: sentiment
    sent = (
        df_hl.groupby(["group", "topic"])["sentiment"]
        .mean()
        .reset_index(name="mean_sentiment")
    )
    sent_pivot = sent.pivot(index="topic", columns="group", values="mean_sentiment").fillna(0)
    sent_pivot.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"], alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Topic-Level Sentiment: High vs Low Ratings")
    axes[1].set_xlabel("Topic ID")
    axes[1].set_ylabel("Mean VADER Compound Score")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Rating Group")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_subgroup_comparison.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} documents")

    docs = df["text_clean"].fillna("").tolist()

    # ── BERTopic ──
    topic_model, topics, probs = run_bertopic(docs)
    print_topic_summary(topic_model)

    # Attach results to dataframe
    df["topic"] = topics
    df["topic_prob"] = [float(p.max()) if hasattr(p, "__len__") else float(p)
                        for p in probs]

    # Save topic label template
    save_topic_labels(topic_model, os.path.join(OUTPUT_DIR, "topic_labels.txt"))

    # ── VADER sentiment ──
    print("\nComputing VADER sentiment scores …")
    df["sentiment"] = compute_vader_sentiment(docs)
    print(f"Mean sentiment: {df['sentiment'].mean():.4f}")

    # ── Aggregate ──
    agg = topic_level_sentiment(df)
    print("\n--- Topic-Level Sentiment ---")
    print(agg.to_string(index=False))

    # ── Save enriched dataframe ──
    df.to_csv(TOPICS_OUT, index=False)
    print(f"\nSaved enriched data → {TOPICS_OUT}")

    # ── Plots ──
    plot_topic_prevalence(df, topic_model)
    plot_topic_sentiment(agg)
    plot_subgroup_comparison(df)

    print("\nDone — all outputs saved to outputs/")


if __name__ == "__main__":
    main()
