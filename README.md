# Assignment 1: Word Embeddings, Sentiments and Topics

**Course:** Large Language Models for Marketing (FEM11154)
**Academic Year:** 2025-2026
**Weight:** 15% | **Deadline:** April 3, 23:59h

## Overview

This project moves from raw marketing text (Yelp reviews) to structured, actionable insights using:
- Word embeddings (Word2Vec)
- Topic modelling (BERTopic)
- Sentiment analysis (VADER)
- Regression analysis (OLS / Logistic)

The dependent variable (DV) is the **star rating** (1–5) from Yelp reviews.

## Dataset

**Source:** [Yelp Academic Dataset](https://www.yelp.com/dataset) (also available on Kaggle as `yelp_review.csv`)

- 6M+ reviews available; we sample **15,000** stratified by star rating
- Columns used: `text` (review body), `stars` (1–5 DV), `business_id`
- Meets all requirements: 5000+ rows, multi-sentence documents, numeric DV

### Download instructions
1. Go to [Kaggle — Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
2. Download `yelp_academic_dataset_review.json` (or `yelp_review.csv`)
3. Place the file in the `data/` folder
4. Run `python scripts/01_data_prep.py` to generate `data/yelp_processed.csv`

## Project Structure

```
llm-marketing-assignment/
├── data/
│   └── yelp_reviews_sample.csv     # processed 15k-row sample
├── scripts/
│   ├── 01_data_prep.py             # load, clean, tokenize
│   ├── 02_embeddings.py            # Word2Vec training + analyses
│   ├── 03_topic_sentiment.py       # BERTopic + VADER
│   └── 04_regression.py            # OLS/logistic regression
├── notebooks/
│   └── analysis.ipynb              # full walkthrough / submission appendix
├── outputs/                        # saved figures and tables
├── requirements.txt
├── plan.md
└── README.md
```

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

Run scripts in order:

```bash
python scripts/01_data_prep.py       # preprocess data
python scripts/02_embeddings.py      # word embeddings
python scripts/03_topic_sentiment.py # topic modelling + sentiment
python scripts/04_regression.py      # DV regression
```

Or open and run `notebooks/analysis.ipynb` end-to-end.

## Assignment Parts Covered

| Part | Description | Script |
|------|-------------|--------|
| 1 | Dataset acquisition & validation | `01_data_prep.py` |
| 2 | Word2Vec embeddings + interpretation | `02_embeddings.py` |
| 3 | *(Dropped in v1.1)* | — |
| 4 | BERTopic + VADER sentiment analysis | `03_topic_sentiment.py` |
| 5 | DV regression with topic/sentiment features | `04_regression.py` |

## Key Libraries

| Task | Library |
|------|---------|
| Word embeddings | `gensim` |
| Topic modelling | `bertopic`, `sentence-transformers` |
| Sentiment | `vaderSentiment` |
| Regression | `statsmodels`, `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
