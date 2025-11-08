"""
Two ways of mixing scores:

- hybrid_rank: reorders content-based candidates using a Bayesian average of
  the ratings, so movies with very few ratings are not overrated.
- blend_scores: mixes collaborative and content scores for every user
  (used in the evaluation and in the "For a user" tab).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def hybrid_rank(content_df: pd.DataFrame, ratings: pd.DataFrame, alpha: float = 0.8) -> pd.DataFrame:
    if content_df.empty:
        return content_df

    agg = ratings.groupby("movieId").agg(
        cnt=("rating", "count"),
        mean=("rating", "mean")
    ).reset_index()

    m = agg["cnt"].quantile(0.60)  # popularity threshold
    C = agg["mean"].mean()         # global mean

    agg["bayes"] = (agg["cnt"]/(agg["cnt"] + m))*agg["mean"] + (m/(agg["cnt"] + m))*C
    qa = agg[["movieId", "bayes"]]

    df = content_df.merge(qa, on="movieId", how="left")
    df["bayes"] = df["bayes"].fillna(C)

    def norm(col: pd.Series) -> pd.Series:
        vmin, vmax = float(col.min()), float(col.max())
        if vmax - vmin < 1e-12:
            return pd.Series([0.0]*len(col), index=col.index)
        return (col - vmin) / (vmax - vmin)

    df["score_n"] = norm(df["score"])
    df["bayes_n"] = norm(df["bayes"])
    df["hybrid"] = alpha*df["score_n"] + (1.0 - alpha)*df["bayes_n"]
    df = df.sort_values("hybrid", ascending=False).reset_index(drop=True)
    return df


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Scales each row (user) to [0, 1] so scores from different models can be mixed."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / np.maximum(hi - lo, 1e-12)


def blend_scores(cf_scores: np.ndarray, content_scores: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """alpha = weight of the collaborative scores, 1 - alpha = weight of the content scores."""
    return alpha * minmax_rows(cf_scores) + (1.0 - alpha) * minmax_rows(content_scores)
