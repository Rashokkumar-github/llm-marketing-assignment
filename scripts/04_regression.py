"""
04_regression.py
----------------
Regression analysis: predict star rating (DV) from text-derived features.

Three models are estimated:
  Model A: stars ~ sentiment_overall                (overall sentiment effect)
  Model B: stars ~ topic_1 ... topic_K              (direct topic effects)
  Model C: stars ~ sentiment + topics + sentiment×topic interactions
                                                     (topic-moderated sentiment)

Variable definitions:
  - sentiment_overall : VADER compound score, mean-centred (μ=0)
  - topic_k_prevalence: BERTopic probability for topic k, standardised (μ=0, σ=1)
  - sentiment×topic_k : product of mean-centred sentiment and standardised topic prob

Input:   data/yelp_topics.csv   (output of 03_topic_sentiment.py)
Output:  outputs/regression_results.txt
         outputs/fig_model_c_coefs.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "yelp_topics.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# Keep only reviews with a clear dominant topic (high topic probability)
MIN_TOPIC_PROB = 0.0    # set > 0 to filter ambiguous docs; 0 keeps all
MAX_TOPICS     = 15     # cap number of topic columns to avoid overfitting


# ─── 1. Feature construction ─────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the regression feature matrix from the enriched topic dataframe.

    Returns a dataframe with columns:
      - stars              (DV, unchanged)
      - sentiment_centered (mean-centred VADER compound)
      - topic_<k>          (standardised BERTopic probability for topic k)
      - sent_x_topic_<k>   (interaction term)
    """
    df = df[df["topic"] != -1].copy()   # drop outlier cluster
    if MIN_TOPIC_PROB > 0:
        df = df[df["topic_prob"] >= MIN_TOPIC_PROB]

    # ── Sentiment: mean-centre ──
    df["sentiment_centered"] = df["sentiment"] - df["sentiment"].mean()

    # ── Topic dummies / probabilities ──
    # Use the hard-assigned topic label as a dummy (1 if doc belongs to topic k)
    # This is clean and interpretable; can be swapped for soft probabilities if desired.
    top_topics = (
        df["topic"].value_counts()
        .head(MAX_TOPICS)
        .index.tolist()
    )

    for k in top_topics:
        col = f"topic_{k}"
        df[col] = (df["topic"] == k).astype(float)

    # Standardise topic dummies to unit variance (facilitates coefficient comparison)
    scaler = StandardScaler()
    topic_cols = [f"topic_{k}" for k in top_topics]
    df[topic_cols] = scaler.fit_transform(df[topic_cols])

    # ── Interaction terms ──
    for k in top_topics:
        df[f"sent_x_topic_{k}"] = df["sentiment_centered"] * df[f"topic_{k}"]

    return df, top_topics


# ─── 2. OLS estimation helper ────────────────────────────────────────────────

def run_ols(y: pd.Series, X: pd.DataFrame, model_name: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with a constant, print summary, return results."""
    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit()
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(model.summary())
    return model


# ─── 3. Plot Model C coefficients ────────────────────────────────────────────

def plot_coefficients(result, title: str, exclude: list[str] = None):
    """
    Plot OLS coefficients with 95% confidence intervals.
    Excludes the intercept and any columns in `exclude`.
    """
    params = result.params.drop("const", errors="ignore")
    conf   = result.conf_int().drop("const", errors="ignore")

    if exclude:
        params = params.drop([c for c in exclude if c in params.index], errors="ignore")
        conf   = conf.drop([c for c in exclude if c in conf.index], errors="ignore")

    # Sort by coefficient magnitude for readability
    order = params.abs().sort_values(ascending=True).index
    params = params[order]
    conf   = conf.loc[order]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in params.values]

    fig, ax = plt.subplots(figsize=(9, max(5, len(params) * 0.35)))
    ax.barh(params.index, params.values, color=colors, alpha=0.8)
    ax.errorbar(
        params.values, params.index,
        xerr=[params.values - conf.iloc[:, 0], conf.iloc[:, 1] - params.values],
        fmt="none", color="black", capsize=3, linewidth=1,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OLS Coefficient")
    ax.set_title(title)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_model_c_coefs.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved coefficient plot → {out}")


# ─── 4. Report to text file ───────────────────────────────────────────────────

def save_report(results: dict, out_path: str):
    """Write model summaries to a plain-text report file."""
    with open(out_path, "w") as f:
        for name, res in results.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"  {name}\n")
            f.write(f"{'='*70}\n")
            f.write(res.summary().as_text())
            f.write("\n")
    print(f"Regression report saved → {out_path}")


# ─── 5. Managerial interpretation ────────────────────────────────────────────

def print_managerial_insights(model_a, model_b, model_c, top_topics: list[int]):
    """
    Print key managerial take-aways from the three models.
    Adapts automatically to the data — no hard-coded topic IDs.
    """
    print("\n--- Managerial Insights ---")

    # Overall sentiment impact (Model A)
    sent_coef = model_a.params.get("sentiment_centered", np.nan)
    sent_p    = model_a.pvalues.get("sentiment_centered", np.nan)
    print(f"\n[Model A] Overall sentiment effect: β = {sent_coef:.3f} (p = {sent_p:.3f})")
    if sent_p < 0.05:
        print("  → Reviewer sentiment is a significant predictor of star ratings.")
    else:
        print("  → Reviewer sentiment alone does not significantly predict star ratings.")

    # Top direct topic effects (Model B)
    topic_cols_b = [c for c in model_b.params.index if c.startswith("topic_")]
    if topic_cols_b:
        top_pos = model_b.params[topic_cols_b].idxmax()
        top_neg = model_b.params[topic_cols_b].idxmin()
        print(f"\n[Model B] Strongest positive topic: {top_pos} (β = {model_b.params[top_pos]:.3f})")
        print(f"[Model B] Strongest negative topic: {top_neg} (β = {model_b.params[top_neg]:.3f})")
        print("  → Topics with positive betas are associated with higher star ratings.")

    # Interaction insights (Model C)
    interact_cols = [c for c in model_c.params.index if c.startswith("sent_x_topic_")]
    if interact_cols:
        sig_interactions = [
            c for c in interact_cols if model_c.pvalues.get(c, 1.0) < 0.05
        ]
        print(f"\n[Model C] Significant sentiment×topic interactions: {len(sig_interactions)}")
        for c in sig_interactions:
            print(f"  {c}: β = {model_c.params[c]:.3f} (p = {model_c.pvalues[c]:.3f})")
        print("  → Positive interactions: sentiment has a stronger effect on rating for this topic.")
        print("  → Negative interactions: the topic dampens the sentiment-rating relationship.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} documents with topic assignments")

    df, top_topics = build_features(df)
    print(f"Feature matrix: {len(df):,} docs, {len(top_topics)} topic features")

    y           = df["stars"]
    sent        = df[["sentiment_centered"]]
    topic_cols  = [f"topic_{k}" for k in top_topics]
    inter_cols  = [f"sent_x_topic_{k}" for k in top_topics]

    # ── Model A: DV ~ sentiment ──
    model_a = run_ols(y, sent, "Model A: stars ~ sentiment_overall")

    # ── Model B: DV ~ topics ──
    model_b = run_ols(y, df[topic_cols], "Model B: stars ~ topic_1 ... topic_K")

    # ── Model C: DV ~ sentiment + topics + interactions ──
    X_c     = pd.concat([sent, df[topic_cols], df[inter_cols]], axis=1)
    model_c = run_ols(y, X_c, "Model C: stars ~ sentiment + topics + sentiment×topics")

    # ── Coefficient plot ──
    plot_coefficients(model_c, "Model C — OLS Coefficients (with 95% CI)")

    # ── Save report ──
    save_report(
        {"Model A": model_a, "Model B": model_b, "Model C": model_c},
        os.path.join(OUTPUT_DIR, "regression_results.txt"),
    )

    # ── Managerial insights ──
    print_managerial_insights(model_a, model_b, model_c, top_topics)

    print("\nDone — all outputs saved to outputs/")


if __name__ == "__main__":
    main()
