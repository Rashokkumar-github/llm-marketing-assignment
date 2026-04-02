# Assignment 1: Word Embeddings, Sentiments, and Topics
**Course:** Large Language Models for Marketing (FEM11154) · Academic Year 2025–2026  
**Dataset:** Yelp Restaurant Reviews (Kaggle) · **N = 10,526 cleaned reviews** (7,003 in regression sample)  
**Dependent Variable:** Star rating (1–5, treated as continuous)

---

## 1. Dataset

The corpus consists of Yelp restaurant reviews drawn from a publicly available Kaggle dataset of 19,896 entries. After stratified sampling (up to 3,000 reviews per star level to avoid class imbalance), HTML stripping, non-ASCII removal, and length filtering (≥ 10 tokens), **10,526 reviews** were retained for embedding and topic modeling. The final regression sample is **7,003 reviews** (after excluding the BERTopic outlier cluster −1). Each document is a multi-sentence free-text evaluation paired with a 1–5 star rating, satisfying all dataset requirements: n ≥ 5,000, multi-sentence structure, and a numeric outcome variable.

---

## 2. Simple Word Embeddings

### 2.1 Training Setup
A **Skip-gram Word2Vec** model (Gensim) was trained directly on the Yelp corpus with embedding dimensionality *d* = 50, window size = 5, minimum token count = 5, and 10 training epochs. Corpus-specific training was preferred over pre-trained GloVe vectors (Wikipedia/Common Crawl) because the Yelp lexicon contains domain-specific vocabulary — cannoli, ghirardelli, bouchon, soft-serve — that is absent or semantically shifted in general corpora. The resulting **embedding matrix is 7,800 × 50**: 7,800 vocabulary words each represented as a 50-dimensional vector. Sample vectors for *food*, *service*, and *price* were printed to verify plausible magnitudes.

### 2.2 Direction Analysis: The Sentiment Axis
A sentiment direction was constructed as the unit-normalised difference vector:

**d**_sentiment = **v**(*good*) − **v**(*bad*)

Every vocabulary word was projected onto this axis via dot product, ranking words from most positive to most negative. The top-15 positive words (*excellent, amazing, fantastic, delicious, love, wonderful, perfect, incredible*) and bottom-15 negative words (*horrible, awful, disgusting, disappointing, terrible, poor, disgusting*) confirm that the learned geometry encodes evaluative polarity without any supervision. This axis is exploited later for document-level sentiment scoring.

![Sentiment Axis](../outputs/fig_sentiment_axis.png)
*Figure 1. Vocabulary projected onto the good − bad direction. Green = positive pole; red = negative pole.*

### 2.3 Dimension Interpretation: Axis 0
Ranking the full vocabulary by their raw value on **dimension 0** reveals a latent axis running from concrete, product-specific nouns at one extreme to abstract, evaluative adjectives at the other. High-scoring words tend to be ingredient or item names (*croissant, ganache, praline*); low-scoring words tend to be generic praise or complaint terms (*great, awful, nice*). This suggests dimension 0 captures **specificity / concreteness** — a property exploited implicitly in the regression (concrete-item topics respond differently to sentiment than generic-praise topics).

### 2.4 Interesting Analysis 1: Word Analogies
The 3CosAdd analogy framework was applied to test semantic compositionality:

| Query | Result |
|---|---|
| *restaurant* − *food* + *hotel* | *lobby, amenities, concierge* |
| *good* − *great* + *bad* | *mediocre, poor* |
| *expensive* − *luxury* + *cheap* | *affordable, budget* |

Results confirm the embedding space supports coherent arithmetic over restaurant-domain concepts, including quality gradations and price positioning — concepts directly relevant to marketing strategy.

### 2.5 Interesting Analysis 2: t-SNE Semantic Neighbourhoods
For five seed words (*service, food, price, ambiance, staff*), the 10 nearest cosine neighbours were retrieved and all 55 words projected to 2D via t-SNE (perplexity = 30, random seed = 42).

![t-SNE Clusters](../outputs/fig_tsne_clusters.png)
*Figure 2. t-SNE projection of semantic neighbourhoods around five marketing-relevant seed words.*

The five clusters are visually distinct, demonstrating that customers encode these experience dimensions as separable conceptual regions. This corroborates the later topic-modeling finding that service incidents, food items, and atmosphere mentions emerge as distinct latent topics.

---

## 3. Topic Modeling and Sentiment Analysis

### 3.1 Document- vs. Sentence-Level Topics
Yelp restaurant reviews describe a **single holistic dining experience**. Although reviewers may mention multiple attributes (food quality, wait time, staff friendliness), these are facets of one overall judgment, not independent topics. Sentence-level BERTopic would fragment semantically coherent evaluations unnecessarily. **Document-level topics** were therefore used.

### 3.2 BERTopic Results
BERTopic was run with the `all-MiniLM-L6-v2` sentence-transformer backend, yielding **66 coherent topics** (plus outlier cluster −1, which captured 3,523 documents with no dominant theme and was excluded from all downstream analyses). BERTopic automatically ranks topics by document count; the 15 largest topics — Topics 0 through 14 — form the analytical focus of this report and serve as predictors in the regression. They were labelled by inspecting each topic's top-8 representative words and exemplar documents:

| ID | Top Words | Label | N docs |
|---|---|---|---|
| 0 | milk, cereal, soft, pie, serve, bar | Soft-Serve & Pastry Counter | 845 |
| 1 | donuts, donut, creme | Donuts | 835 |
| 2 | macarons, bouchon, venetian, bakery, croissant | French Bakery & Macarons | 638 |
| 3 | ice, cream, flavors | Ice Cream | 616 |
| 4 | cupcake, cupcakes, sprinkles, velvet, red | Cupcakes | 568 |
| 5 | cake, cakes, slice, birthday | Custom Cakes & Birthdays | 286 |
| 6 | amelie, french, charlotte | French Café (Amelie) | 257 |
| 7 | she, he, we, me, us | Personal Narratives / Service Incidents | 236 |
| 8 | coffee, place, great, food | Coffee & General Praise | 219 |
| 9 | lobster, tail, tails, cannoli | Lobster Tail Cannoli | 216 |
| 10 | presti, italian, italy, pizza | Italian Bakery (Presti's) | 139 |
| 11 | cannoli, cannolis, bakery | Cannoli | 120 |
| 12 | charlotte, place, location | Charlotte / Location Mentions | 119 |
| 13 | croissant, croissants, almond, cheese | Croissants & Almond Pastries | 99 |
| 14 | sundae, ghirardelli, fudge, hot, chocolate | Hot Fudge Sundaes (Ghirardelli) | 97 |

The remaining 51 topics (Topics 15–65) are excluded from downstream analyses. Because BERTopic automatically ranks topics by document count, Topics 0–14 are simply the 15 largest, and limiting the regression to 15 predictors keeps the model interpretable without arbitrary feature selection. No universal sample-size threshold exists for this decision; the cutoff is a deliberate scope choice, not a statistical rule.

### 3.3 Sentiment Scores (VADER)
VADER compound scores (range −1 to +1) were computed per document and aggregated to the topic level for the **15 focal topics only**. The **corpus-level mean across these topics is +0.316** — moderately positive, consistent with the publication-selection bias of online reviews where satisfied customers outnumber vocal detractors.

![Topic Sentiment](../outputs/fig_topic_sentiment.png)
*Figure 3. Mean VADER compound score for the 15 focal topics, sorted from lowest to highest. Green = net positive; red = net negative.*

All 14 product-and-place topics are net positive; **Topic 7 (Personal Narratives) is the sole net-negative topic** at −0.082. Three patterns emerge from the ranking:

- **Location and general-praise topics score highest** (Topics 12, 10, 8, 6 all above +0.78). Reviewers writing about place, atmosphere, or an Italian neighbourhood bakery tend toward effusive, validating language — these are "I love this place" reviews rather than evaluative ones.
- **Bespoke and premium-specialty items score lowest among the positive topics** — custom cakes (+0.541), cannoli (+0.394), sundaes (+0.516), and lobster-tail cannoli (+0.593). Despite a positive compound score, the gap relative to the top-scoring topics reflects a higher incidence of disappointed expectations: customers who seek out a specialty item and find it below expectations write more negatively than customers making routine purchases.
- **Topic 7 is categorically different from all others**. Its net-negative score (−0.082) is not just the lowest — it crosses the zero line. The pronoun-heavy vocabulary (*she, he, we, me, us*) marks reviews structured as interpersonal incident accounts (staff rudeness, billing errors, long-wait narratives). No product-specific topic produces this pattern; complaint escalation is interpersonal, not product-led.

### 3.4 Subgroup Comparison: High- vs. Low-Rating Reviewers
Reviewers were split into **high-raters (4–5★)** and **low-raters (1–2★)** and compared on both topic prevalence and topic-level sentiment.

![Subgroup Comparison](../outputs/fig_subgroup_comparison.png)
*Figure 4. Left: topic prevalence (share of reviews) across rating groups. Right: mean sentiment per topic across groups.*

**Topic prevalence:** High-raters disproportionately discuss ice cream (Topic 3), coffee and general praise (Topic 8), and location atmosphere (Topic 12). Low-raters are heavily concentrated in the **Personal Narratives topic (Topic 7)** — reviews structured around a specific service incident involving named staff or third parties.

**Topic-level sentiment:** Across nearly all topics, high-raters score substantially higher on VADER. The gap is widest for Topic 7 (service narratives) and Topic 5 (custom cakes), suggesting these categories are most sensitive to execution failure.

**Marketing interpretation:** The divergence in Topic 7 prevalence between rating groups is the single most actionable pattern in the data. Dissatisfied customers do not simply rate food lower — they write *narratives* about specific interactions. This implies that **complaint escalation is interpersonal, not product-led**: managing front-of-house staff behaviour should be prioritised by restaurants seeking to reduce 1–2★ reviews.

---

## 4. Regression Analysis: Drivers of Star Ratings

Three nested OLS models were estimated with star rating as the dependent variable.

**Variable definitions:**
- `sentiment_centered` = VADER compound score minus its corpus mean (μ = 0); preserves the unit-per-star interpretation.
- `topic_k` = binary indicator for hard-assigned topic *k*, standardised to μ = 0, σ = 1 via StandardScaler; enables coefficient comparison across topics.
- `sent_x_topic_k` = product of `sentiment_centered` × `topic_k`; captures whether sentiment's predictive power is amplified or dampened within topic *k*.

### 4.1 Model A — Sentiment Only

| | Coefficient | SE | *p* |
|---|---|---|---|
| Intercept | 3.475 | 0.013 | < .001 |
| sentiment_centered | **1.353** | 0.025 | < .001 |

**R² = 0.293**, F(1, 7001) = 2,905, *p* < .001.

Sentiment alone explains **29.3% of rating variance**. The coefficient β = 1.353 implies that moving from the most negative VADER score (−1) to the most positive (+1) is associated with a **2.7-star swing** — spanning more than half the five-point scale. Sentiment is the dominant single predictor of how customers rate a restaurant.

### 4.2 Model B — Topic Effects Only

| Topic | Coefficient | *p* | Label |
|---|---|---|---|
| Topic 7 | −0.365 | < .001 | Personal Narratives |
| Topic 0 | −0.192 | < .001 | Soft-Serve & Pastry Counter |
| Topic 5 | −0.130 | < .001 | Custom Cakes |
| Topic 4 | −0.111 | < .001 | Cupcakes |
| Topic 9 | −0.113 | < .001 | Lobster Tail Cannoli |
| Topic 11 | −0.113 | < .001 | Cannoli |
| Topic 3 | +0.078 | < .001 | Ice Cream |
| Topic 12 | +0.071 | < .001 | Location Mentions |
| Topic 8 | +0.050 | .001 | Coffee & General Praise |
| Topic 1 | +0.037 | .027 | Donuts |

**R² = 0.127**, F(15, 6987) = 67.92, *p* < .001. Topics alone explain 12.7% of variance — less than sentiment, but the pattern is substantively important: bespoke and premium-positioned items (custom cakes, specialty cannoli, cupcakes) carry **structural negative coefficients** even before controlling for sentiment.

### 4.3 Model C — Full Model (Sentiment + Topics + Interactions)

**R² = 0.363**, F(31, 6971) = 128.2, *p* < .001. Adding topics and interactions raises explained variance by **+7.0 percentage points** (+24% relative improvement) over Model A.

| Term | Coefficient | *p* |
|---|---|---|
| sentiment_centered | **1.278** | < .001 |
| Topic 7 (Narratives) | −0.318 | < .001 |
| Topic 0 (Soft-Serve) | −0.181 | < .001 |
| Topic 5 (Cakes) | −0.100 | < .001 |
| sent × Topic 7 | **−0.216** | < .001 |
| sent × Topic 0 | −0.179 | < .001 |
| sent × Topic 9 (Cannoli) | −0.101 | < .001 |
| sent × Topic 11 (Cannoli) | −0.091 | < .001 |
| sent × Topic 2 (French Bakery) | −0.121 | < .001 |
| sent × Topic 4 (Cupcakes) | −0.091 | .001 |
| sent × Topic 6 (French Café) | −0.079 | .009 |

All statistically significant interaction terms are **negative**, meaning sentiment's predictive leverage on ratings is *dampened* within every identified topic relative to the unmodelled baseline. The net sentiment effect for a review in Topic 7 is 1.278 − 0.216 = **1.062** (a 17% reduction); for Topic 0 it is 1.278 − 0.179 = **1.099** (14% reduction).

**Interpretation:** In generic or celebratory reviews (Topic 8, Topic 3), positive sentiment maps reliably to high ratings because there is little else to anchor the evaluation. In narrative and product-specific reviews, the *content* of the claim — a specific service incident, a stale croissant, a poorly iced cake — overrides emotional framing. Positive language cannot compensate for a bad specific experience.

![Model C Coefficients](../outputs/fig_model_c_coefs.png)
*Figure 5. OLS Model C: all 31 coefficients with 95% confidence intervals. Green = positive effect on star rating; red = negative.*

---

## 5. Managerial Implications

**1. Build topic-segmented dashboards, not single sentiment scores.**  
A single VADER aggregate score explains 29% of rating variance; adding topic structure raises this to 36%. A dashboard that reports sentiment *within* each topic (ice cream vs. custom cakes vs. service narratives) is significantly more informative for operational decisions than a single headline number.

**2. Service incidents are the primary driver of 1–2★ reviews.**  
Topic 7 (Personal Narratives) is the strongest negative predictor across all three models (β = −0.365 in Model B; β = −0.318 in Model C), and its sentiment-dampening interaction (β = −0.216) is the largest of all 15. Dissatisfied customers write *stories* — they describe specific staff members, recall exact dialogue, and use personal pronouns. Restaurants should flag high-pronoun reviews for **manual triage and direct service recovery**, not automated sentiment tracking.

**3. Premium/bespoke items create expectation risk.**  
Custom cakes (Topic 5), cupcakes (Topic 4), specialty cannoli (Topics 9, 11), and the pastry counter (Topic 0) all carry structural negative coefficients even controlling for sentiment. Customers approach these categories with high expectations; any quality or execution gap is disproportionately penalised. **Expectation calibration** (transparent lead times, explicit ingredient sourcing, in-store quality guarantees) and tighter QA protocols in these categories would yield the highest marginal return in ratings.

**4. Shift monitoring from sentiment to claim-tracking for specialty products.**  
The large interaction dampening for French Bakery (Topic 2), cannoli (Topics 9, 11), and croissants (Topic 13) means that positive emotional language in these reviews does not predict high ratings as reliably as in generic categories. These customers evaluate on **specific quality assertions** — freshness, texture, layering — rather than mood. Monitoring systems for these categories should extract and track factual quality claims, not just tone.

**5. Ice cream and coffee are reputational assets.**  
Topics 3 and 8 carry positive coefficients and their sentiment interactions are non-significant (β ≈ 0), meaning positive sentiment in these categories converts reliably into high ratings. These categories should be treated as **halo products**: investing in their quality and visibility is likely to lift overall venue scores.

---

## References

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.

Hutto, C. J., & Gilbert, E. E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of ICWSM*.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems, 26*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP*.

---

*Appendix: Clean, reproducible code is provided in `notebooks/analysis.ipynb` and the four modular scripts in `scripts/` (`01_data_prep.py`, `02_embeddings.py`, `03_topic_sentiment.py`, `04_regression.py`).*
