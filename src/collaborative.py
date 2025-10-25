"""
Collaborative filtering models for top-N recommendation.

All models work on a users x movies ratings matrix (scipy sparse, 0 = not rated)
and return a dense users x movies matrix of scores.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class Popularity:
    """Baseline: recommends the most rated movies to everyone."""

    def fit(self, R):
        self.counts = np.asarray((R > 0).sum(axis=0)).ravel().astype(float)
        return self

    def score_users(self, R) -> np.ndarray:
        return np.tile(self.counts, (R.shape[0], 1))


class ItemKNN:
    """
    Item-based CF with cosine similarity between the rating columns.
    score(user, item) = sum over rated items j of rating(user, j) * sim(j, item)
    """

    def fit(self, R):
        self.X = normalize(R.astype(float), axis=0)
        return self

    def score_users(self, R) -> np.ndarray:
        # (R @ X.T) @ X is the same as R @ (X.T @ X) but never builds the
        # full item x item similarity matrix
        scores = (R @ self.X.T) @ self.X
        return scores.toarray() if hasattr(scores, "toarray") else np.asarray(scores)


class PureSVD:
    """
    PureSVD (Cremonesi et al., 2010): truncated SVD of the ratings matrix with
    missing values as 0. Scores are R @ V @ V.T, so a user's scores are their
    ratings projected on the k latent factors.
    """

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, R):
        _, _, vt = svds(R.astype(float), k=self.k)
        self.V = vt.T
        return self

    def score_users(self, R) -> np.ndarray:
        return np.asarray((R @ self.V) @ self.V.T)
