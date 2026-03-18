# Assignment 1: Implementation Plan

## Context
Course assignment (FEM11154, due April 3) worth 15%. Goal: move from raw marketing text to structured insights using word embeddings, topic modelling, sentiment analysis, and regression. Dataset: Yelp Academic Dataset (reviews with 1–5 star ratings as the DV). Part 3 (contextual embeddings) is dropped per v1.1.

---

## Step-by-Step Plan

### Step 0 — Setup ✅
- Init git repo locally, create GitHub remote, push initial commit
- `requirements.txt`, `README.md`, `plan.md` created

### Step 1 — Dataset (Part 1)
**Script:** `scripts/01_data_prep.py`

- Source: Yelp Academic Dataset (Kaggle `yelp_review.csv`) — sample 15k rows stratified by star rating
- Columns: `text`, `stars` (DV), `business_id`
- Clean: lowercase, strip HTML/special chars
- Save two versions:
  - `data/yelp_processed.csv` — cleaned text + metadata
  - Tokenized sentences stored in-memory for Word2Vec

**Dataset criteria check:**
- ≥ 5,000 observations ✓
- Multi-sentence documents ✓ (Yelp reviews average ~3–5 sentences)
- Numeric DV (star rating 1–5) ✓

---

### Step 2 — Word Embeddings (Part 2)
**Script:** `scripts/02_embeddings.py`

- Train Word2Vec (Skip-gram) via `gensim`:
  - `vector_size=50`, `window=5`, `min_count=5`, `sg=1`
- Print embedding matrix shape + 3 sample vectors

**Direction analysis:**
- Define sentiment direction: `vec('good') - vec('bad')`
- Project all vocabulary onto this axis via dot product
- Print top-10 and bottom-10 words → interprets the direction as a sentiment axis

**Dimension analysis:**
- Pick Dimension 0; sort all vocab by that value
- Identify its latent meaning from the extremes

**Two interesting analyses:**
1. **Word analogies** — `vec('restaurant') - vec('food') + vec('hotel') ≈ ?` (king-queen style)
2. **Cosine similarity clusters** — nearest neighbours for ["service", "price", "food", "ambiance"]; visualize with t-SNE/PCA

---

### Step 3 — Topic Modelling & Sentiment (Part 4)
**Script:** `scripts/03_topic_sentiment.py`

- **Rationale for document-level topics**: Yelp reviews typically cover one overall experience → use document-level BERTopic
- **BERTopic setup**: `sentence-transformers` backend (`all-MiniLM-L6-v2`)
- Manually label each topic using top words + 3 exemplar documents (e.g., "Delivery Delays", "Ambiance & Atmosphere", "Value for Money")
- **VADER** sentiment: `compound` score per document
- Aggregate to topic level: mean compound score per topic

**Subgroup comparison — High (4–5★) vs Low (1–2★):**
- Topic prevalence across groups (bar chart)
- Topic-level sentiment across groups (grouped bar chart)
- Substantive marketing interpretation

---

### Step 4 — DV Regression (Part 5)
**Script:** `scripts/04_regression.py`

**DV:** `stars` (1–5, treated as continuous for OLS)

**Features:**
- `sentiment_overall` — VADER compound (mean-centered)
- `topic_k_prevalence` — BERTopic topic probability for topic k (standardized, unit variance)
- `sentiment_x_topic_k` — interaction: overall sentiment × topic k prevalence

**Model sequence:**
| Model | Formula | Purpose |
|-------|---------|---------|
| A | `stars ~ sentiment_overall` | Overall sentiment impact |
| B | `stars ~ topic_1 + ... + topic_K` | Direct topic effects |
| C | `stars ~ sentiment + topics + sentiment×topics` | Topic-moderated sentiment effects |

Report: coefficients, p-values, R², managerial implications

---

### Step 5 — Notebook & Report
- `notebooks/analysis.ipynb`: runs all scripts sequentially, adds narrative markdown
- Written report (5 pages max): methodology → embeddings → topics/sentiment → regression → implications

---

## Commit Strategy
| Commit message | Contents |
|---------------|----------|
| `init: project structure, README, requirements` | scaffolding |
| `data: preprocessing script` | `01_data_prep.py` |
| `embeddings: word2vec training and analyses` | `02_embeddings.py` |
| `topics: bertopic + sentiment analysis` | `03_topic_sentiment.py` |
| `regression: DV driver analysis` | `04_regression.py` |
| `notebook: full analysis notebook` | `analysis.ipynb` |

---

## Verification Checklist
- [ ] Each script runs standalone without errors
- [ ] Figures saved to `outputs/`
- [ ] Notebook runs top-to-bottom cleanly
- [ ] All assignment parts covered: Part 1 ✓, Part 2 ✓, Part 3 dropped ✓, Part 4 ✓, Part 5 ✓
