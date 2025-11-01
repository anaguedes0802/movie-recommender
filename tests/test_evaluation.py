import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import temporal_split, top_k, precision_recall_ndcg, liked_weights
from src.hybrid_recommender import minmax_rows


def test_temporal_split_keeps_latest_ratings_for_test():
    ratings = pd.DataFrame({
        "userId": [1] * 10,
        "movieId": range(10),
        "rating": [4.0] * 10,
        "timestamp": range(100, 110),
    })
    train, test = temporal_split(ratings, test_frac=0.2)
    assert len(test) == 2
    assert test["timestamp"].min() > train["timestamp"].max()


def test_top_k_skips_seen_movies():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    seen = csr_matrix(np.array([[1, 0, 0, 0]]))
    recs = top_k(scores, seen, k=2)
    assert recs.tolist() == [[1, 3]]


def test_perfect_ranking_has_ndcg_one():
    recs = np.array([[0, 1, 2]])
    res = precision_recall_ndcg(recs, [{0, 1}], k=3)
    assert res["recall@3"] == 1.0
    assert abs(res["ndcg@3"] - 1.0) < 1e-9


def test_liked_weights_falls_back_to_all_ratings():
    R = csr_matrix(np.array([[5.0, 2.0], [3.0, 1.0]]))
    W = liked_weights(R).toarray()
    assert W[0].tolist() == [2.0, 0.0]
    assert W[1].tolist() == [3.0, 1.0]  # no movie >= 4, keeps everything


def test_minmax_rows():
    out = minmax_rows(np.array([[1.0, 3.0, 2.0]]))
    assert out.tolist() == [[0.0, 1.0, 0.5]]
