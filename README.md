# Assignment 1: Word Embeddings, Sentiments and Topics

**Course:** Large Language Models for Marketing (FEM11154)
**Academic Year:** 2025-2026
**Weight:** 15% | **Deadline:** April 3, 23:59h

## Overview

This project moves from raw marketing text (Yelp restaurant reviews) to structured, actionable insights using:
- Word embeddings (Word2Vec)
- Topic modelling (BERTopic)
- Sentiment analysis (VADER)
- Regression analysis (OLS)

The dependent variable (DV) is the **star rating** (1–5).

## Dataset

**File:** `data/Yelp Restaurant Reviews.csv` (included in this repo)

| Property | Detail |
|----------|--------|
| Raw rows | 19,896 reviews |
| Cleaned sample | 10,526 reviews (stratified, ≥ 10 tokens, HTML stripped) |
| Regression sample | 7,003 reviews (BERTopic outlier cluster −1 excluded) |
| Columns | `Yelp URL`, `Rating` (1–5), `Date`, `Review Text` |
| DV | `Rating` — star rating 1–5 |
| Source | Yelp restaurant reviews |

The dataset meets all assignment criteria:
- ≥ 5,000 observations ✓
- Multi-sentence documents ✓
- Numeric dependent variable (star rating 1–5) ✓

> **Kaggle version:** This dataset is also publicly available on Kaggle at
> [farukalam/yelp-restaurant-reviews](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews).
> No download is needed here — the CSV is already included in `data/`.

## Project Structure

```
├── data/
│   ├── Yelp Restaurant Reviews.csv   # raw dataset (19,896 rows)
│   ├── yelp_processed.csv            # cleaned 10,526-review sample (01_data_prep.py)
│   └── yelp_topics.csv               # enriched with topic + sentiment columns (03_topic_sentiment.py)
├── docs/
│   └── assignment.md                 # full written report (all parts)
├── scripts/
│   ├── 01_data_prep.py               # load, clean, tokenise
│   ├── 02_embeddings.py              # Word2Vec training, sentiment axis, word analogies, t-SNE
│   ├── 03_topic_sentiment.py         # BERTopic + VADER sentiment + subgroup comparison
│   └── 04_regression.py              # OLS regression (DV drivers)
├── notebooks/
│   └── analysis.ipynb                # full walkthrough / submission appendix
├── outputs/
│   ├── word2vec_model.bin            # trained Word2Vec model
│   ├── fig_sentiment_axis.png        # vocabulary projected onto good−bad direction
│   ├── fig_tsne_clusters.png         # t-SNE semantic neighbourhoods (5 seed words)
│   ├── fig_topic_prevalence.png      # BERTopic document counts (top 20 topics)
│   ├── fig_topic_sentiment.png       # mean VADER score for 15 focal topics
│   ├── fig_subgroup_comparison.png   # high- vs low-rater topic prevalence & sentiment
│   ├── fig_model_c_coefs.png         # OLS Model C coefficients with 95% CIs
│   ├── topic_labels.txt              # top-8 words per topic (manual label template)
│   └── regression_results.txt        # full OLS output (Models A, B, C)
├── requirements.txt
├── plan.md
└── README.md
```

## Setup

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

Run scripts in order:

```bash
python3 scripts/01_data_prep.py        # preprocess → data/yelp_processed.csv
python3 scripts/02_embeddings.py       # word embeddings → outputs/
python3 scripts/03_topic_sentiment.py  # topic modelling + sentiment → outputs/
python3 scripts/04_regression.py       # DV regression → outputs/
```

Or open and run `notebooks/analysis.ipynb` end-to-end.

## Assignment Parts Covered

| Part | Description | Script |
|------|-------------|--------|
| 1 | Dataset acquisition & validation | `01_data_prep.py` |
| 2 | Word2Vec: sentiment axis, dimension analysis, word analogies, t-SNE clusters | `02_embeddings.py` |
| 3 | BERTopic + VADER sentiment + high/low-rater subgroup comparison | `03_topic_sentiment.py` |
| 4 | OLS regression: sentiment + topic effects + interactions | `04_regression.py` |
| 5 | Managerial implications | `docs/assignment.md` |

## Key Libraries

| Task | Library |
|------|---------|
| Word embeddings | `gensim` |
| Topic modelling | `bertopic`, `sentence-transformers` |
| Sentiment | `vaderSentiment` |
| Regression | `statsmodels`, `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
