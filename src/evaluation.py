"""
Offline evaluation for top-N recommendation.

- Split: per user, the most recent 20% of the ratings go to the test set.
- A test movie counts as relevant if the user rated it 4.0 or more.
- Movies already rated in train are never recommended.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    ratings = ratings.sort_values(["userId", "timestamp"])
    # 0 = most recent rating of the user
    recency = ratings.groupby("userId").cumcount(ascending=False)
    n = ratings.groupby("userId")["movieId"].transform("size")
    n_test = np.maximum(1, np.round(n * test_frac)).astype(int)
    is_test = recency < n_test
    return ratings[~is_test].copy(), ratings[is_test].copy()


def ratings_matrix(ratings: pd.DataFrame, user_ids, movie_ids) -> csr_matrix:
    u = pd.Index(user_ids).get_indexer(ratings["userId"])
    m = pd.Index(movie_ids).get_indexer(ratings["movieId"])
    keep = (u >= 0) & (m >= 0)
    return csr_matrix(
        (ratings["rating"].values[keep], (u[keep], m[keep])),
        shape=(len(user_ids), len(movie_ids)),
    )


def liked_weights(R: csr_matrix, threshold: float = 4.0) -> csr_matrix:
    """
    User profile weights for the content models: rating - 3 for movies rated
    >= threshold (so 4 -> 1, 5 -> 2). Users without any liked movie keep all
    their ratings.
    """
    W = R.copy().astype(float)
    W.data = np.where(W.data >= threshold, W.data - 3.0, 0.0)
    W.eliminate_zeros()
    empty = np.asarray(W.sum(axis=1)).ravel() == 0
    if empty.any():
        W = (W + R.multiply(empty[:, None])).tocsr()
    return W


def top_k(scores: np.ndarray, seen: csr_matrix, k: int = 10) -> np.ndarray:
    scores = scores.astype(float, copy=True)
    scores[seen.nonzero()] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    # argpartition doesn't sort, so sort the k best
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def relevant_sets(test: pd.DataFrame, user_ids, movie_ids, threshold: float = 4.0) -> list[set]:
    rel = test[test["rating"] >= threshold]
    u = pd.Index(user_ids).get_indexer(rel["userId"])
    m = pd.Index(movie_ids).get_indexer(rel["movieId"])
    sets = [set() for _ in user_ids]
    for ui, mi in zip(u, m):
        if ui >= 0 and mi >= 0:
            sets[ui].add(mi)
    return sets


def precision_recall_ndcg(recs: np.ndarray, relevant: list[set], k: int = 10) -> dict:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    prec, rec, ndcg = [], [], []
    for row, rel in zip(recs, relevant):
        if not rel:
            continue
        hits = np.array([i in rel for i in row[:k]], dtype=float)
        prec.append(hits.sum() / k)
        rec.append(hits.sum() / len(rel))
        ideal = discounts[: min(len(rel), k)].sum()
        ndcg.append((hits * discounts).sum() / ideal)
    return {
        f"precision@{k}": float(np.mean(prec)),
        f"recall@{k}": float(np.mean(rec)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        "users": len(prec),
    }


def coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that is recommended to at least one user."""
    return len(np.unique(recs)) / n_items
