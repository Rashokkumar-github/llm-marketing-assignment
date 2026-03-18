"""
02_embeddings.py
----------------
Train Word2Vec embeddings on the processed Yelp corpus, then run:
  - Embedding matrix inspection
  - Sentiment direction analysis (Part 2 – direction)
  - Single dimension interpretation (Part 2 – axis)
  - Two interesting analyses:
      1. Word analogies (king-queen style)
      2. Cosine-similarity cluster visualisation (t-SNE)

Input:   data/yelp_processed.csv
Outputs: outputs/word2vec_model.bin
         outputs/fig_tsne_clusters.png
         outputs/fig_sentiment_axis.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "yelp_processed.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

VECTOR_SIZE = 50
WINDOW      = 5
MIN_COUNT   = 5
EPOCHS      = 10
SEED        = 42


# ─── 1. Train Word2Vec ────────────────────────────────────────────────────────

def train_word2vec(df: pd.DataFrame) -> Word2Vec:
    """Train a Skip-gram Word2Vec model on the tokenised corpus."""
    sentences = [row.split() for row in df["tokens"].dropna()]
    print(f"Training Word2Vec on {len(sentences):,} documents …")
    model = Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,           # Skip-gram
        epochs=EPOCHS,
        seed=SEED,
        workers=4,
    )
    print(f"Vocabulary size: {len(model.wv):,} words")
    return model


def inspect_model(model: Word2Vec):
    """Print embedding matrix shape and a few sample vectors."""
    wv = model.wv
    print(f"\n--- Embedding Matrix ---")
    print(f"Shape: {wv.vectors.shape}  (vocab_size × vector_size)")
    sample_words = ["food", "service", "price"]
    for w in sample_words:
        if w in wv:
            print(f"\nVector for '{w}' (first 10 dims):\n  {wv[w][:10]}")


# ─── 2. Sentiment direction ───────────────────────────────────────────────────

def sentiment_direction_analysis(model: Word2Vec, top_n: int = 15) -> np.ndarray:
    """
    Define a sentiment direction as vec('good') - vec('bad').
    Project the entire vocabulary onto this axis and rank words.
    Returns the direction vector.
    """
    wv = model.wv
    if "good" not in wv or "bad" not in wv:
        print("Warning: 'good' or 'bad' not in vocabulary. Skipping direction analysis.")
        return None

    direction = wv["good"] - wv["bad"]
    direction = direction / np.linalg.norm(direction)   # unit vector

    # Project all words
    scores = {word: float(np.dot(wv[word], direction)) for word in wv.key_to_index}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n--- Sentiment Direction (vec('good') - vec('bad')) ---")
    print(f"Top {top_n} words (most positive):")
    for w, s in sorted_scores[:top_n]:
        print(f"  {w:20s}  {s:.4f}")
    print(f"\nBottom {top_n} words (most negative):")
    for w, s in sorted_scores[-top_n:]:
        print(f"  {w:20s}  {s:.4f}")

    # Plot
    top_words    = sorted_scores[:top_n]
    bottom_words = sorted_scores[-top_n:]
    plot_words   = top_words + bottom_words
    labels = [w for w, _ in plot_words]
    values = [s for _, s in plot_words]
    colors = ["#2ecc71"] * top_n + ["#e74c3c"] * top_n

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Projection onto sentiment axis")
    ax.set_title("Sentiment Direction: vec('good') − vec('bad')")
    ax.invert_yaxis()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_sentiment_axis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved sentiment axis plot → {out}")

    return direction


# ─── 3. Single dimension interpretation ──────────────────────────────────────

def dimension_analysis(model: Word2Vec, dim: int = 0, top_n: int = 15):
    """
    Inspect a single embedding dimension by ranking vocabulary words by their value on that axis.
    """
    wv = model.wv
    scores = {word: float(wv[word][dim]) for word in wv.key_to_index}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n--- Dimension {dim} Analysis ---")
    print(f"Top {top_n} words (highest value on dim {dim}):")
    for w, s in sorted_scores[:top_n]:
        print(f"  {w:20s}  {s:.4f}")
    print(f"\nBottom {top_n} words (lowest value on dim {dim}):")
    for w, s in sorted_scores[-top_n:]:
        print(f"  {w:20s}  {s:.4f}")
    print(
        f"\nInterpretation hint: examine the contrast between the top and bottom word lists "
        f"to infer the latent concept encoded in dimension {dim}."
    )


# ─── 4a. Word analogies ───────────────────────────────────────────────────────

def word_analogies(model: Word2Vec):
    """
    Run several word analogy queries using the 3CosAdd method:
    result ≈ vec(B) - vec(A) + vec(C)
    """
    wv = model.wv
    queries = [
        ("restaurant", "food",    "hotel",   "What does a hotel serve?"),
        ("good",       "great",   "bad",      "Antonym of great?"),
        ("expensive",  "luxury",  "cheap",    "Cheap luxury?"),
    ]

    print("\n--- Word Analogies ---")
    for a, b, c, label in queries:
        if not all(w in wv for w in [a, b, c]):
            print(f"  Skipping '{label}': one or more words not in vocab")
            continue
        results = wv.most_similar(positive=[b, c], negative=[a], topn=3)
        words = [r[0] for r in results]
        print(f"  {b} - {a} + {c} ≈ {words}   ({label})")


# ─── 4b. Cosine-similarity cluster visualisation ─────────────────────────────

def tsne_cluster_plot(model: Word2Vec, seed_words: list[str], n_neighbors: int = 10):
    """
    For each seed word, gather its nearest neighbours then project all of them
    to 2-D using t-SNE and plot a colour-coded scatter.
    """
    wv = model.wv
    all_words   = []
    word_groups = {}

    for seed in seed_words:
        if seed not in wv:
            print(f"  '{seed}' not in vocabulary, skipping.")
            continue
        neighbours = [seed] + [w for w, _ in wv.most_similar(seed, topn=n_neighbors)]
        word_groups[seed] = neighbours
        all_words.extend(neighbours)

    all_words = list(dict.fromkeys(all_words))   # deduplicate, preserve order
    if len(all_words) < 5:
        print("Not enough words for t-SNE plot.")
        return

    vectors = np.array([wv[w] for w in all_words])

    # t-SNE
    perplexity = min(30, len(all_words) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=1000)
    coords = tsne.fit_transform(vectors)

    # Assign group colour
    palette = sns.color_palette("tab10", n_colors=len(word_groups))
    group_map = {}
    for i, (seed, words) in enumerate(word_groups.items()):
        for w in words:
            group_map[w] = (seed, palette[i])

    fig, ax = plt.subplots(figsize=(11, 8))
    for i, word in enumerate(all_words):
        seed_label, color = group_map.get(word, ("other", "grey"))
        ax.scatter(coords[i, 0], coords[i, 1], color=color, s=40, alpha=0.8)
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.85)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i],
               markersize=8, label=seed)
        for i, seed in enumerate(word_groups.keys())
    ]
    ax.legend(handles=legend_elements, title="Seed word", loc="best")
    ax.set_title("t-SNE: Word Embedding Clusters")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_tsne_clusters.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved t-SNE cluster plot → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} documents")

    # Train
    model = train_word2vec(df)
    model.save(os.path.join(OUTPUT_DIR, "word2vec_model.bin"))
    print(f"Model saved → {OUTPUT_DIR}/word2vec_model.bin")

    # Inspect
    inspect_model(model)

    # Direction analysis
    sentiment_direction_analysis(model)

    # Dimension analysis
    dimension_analysis(model, dim=0)

    # Interesting analysis 1: analogies
    word_analogies(model)

    # Interesting analysis 2: t-SNE cluster visualisation
    seed_words = ["service", "food", "price", "ambiance", "staff"]
    tsne_cluster_plot(model, seed_words, n_neighbors=10)

    print("\nDone — all outputs saved to outputs/")


if __name__ == "__main__":
    main()
