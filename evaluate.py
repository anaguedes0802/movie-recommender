"""
Offline evaluation of all the models on MovieLens small.

Run with:
    python evaluate.py

Saves the results to results/ (metrics.csv, tuning.csv and metrics.png).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.prepare_data import load_data
from src.content_model import ContentRecommender
from src.embeddings import load_or_build_embeddings, with_genres
from src.collaborative import Popularity, ItemKNN, PureSVD
from src.hybrid_recommender import blend_scores
from src.evaluation import (
    temporal_split, ratings_matrix, liked_weights, top_k,
    relevant_sets, precision_recall_ndcg, coverage,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

K = 10
SVD_FACTORS = [8, 16, 32, 64, 128]
ALPHAS = [round(a, 1) for a in np.arange(0, 1.01, 0.1)]
HYBRID_CONTENT = ["Embeddings", "Embeddings + genres"]


def evaluate_scores(scores, R_train, relevant, cold_relevant, cold_mask, n_items):
    recs = top_k(scores, R_train, K)
    res = precision_recall_ndcg(recs, relevant, K)
    res["coverage"] = coverage(recs, n_items)
    # cold-start check: rank only the movies with no ratings in train and see
    # if the ones the user liked in test come up
    cold_scores = scores.astype(float, copy=True)
    cold_scores[:, ~cold_mask] = -np.inf
    cold_recs = top_k(cold_scores, R_train, K)
    res[f"recall@{K} (cold-start)"] = precision_recall_ndcg(cold_recs, cold_relevant, K)[f"recall@{K}"]
    return res


def run(train, test, movies, content_models, svd_k=None, content=None, alpha=None):
    """Fits/scores every model on train and evaluates on test. Returns a dict model -> metrics."""
    user_ids = np.sort(train["userId"].unique())
    movie_ids = movies["movieId"].values
    R = ratings_matrix(train, user_ids, movie_ids)
    W = liked_weights(R)

    relevant = relevant_sets(test, user_ids, movie_ids)
    train_counts = np.asarray((R > 0).sum(axis=0)).ravel()
    cold_mask = train_counts == 0
    cold = set(np.where(cold_mask)[0])
    cold_relevant = [r & cold for r in relevant]

    scores = {}
    scores["Popularity"] = Popularity().fit(R).score_users(R)
    for name, model in content_models.items():
        scores[name] = model.score_users(W)
    scores["ItemKNN"] = ItemKNN().fit(R).score_users(R)

    results = {}
    if svd_k is None:
        # tuning mode: try every k, both embedding models and every alpha
        for k in SVD_FACTORS:
            svd_scores = PureSVD(k).fit(R).score_users(R)
            results[f"PureSVD k={k}"] = evaluate_scores(svd_scores, R, relevant, cold_relevant, cold_mask, len(movie_ids))
            for c in HYBRID_CONTENT:
                for a in ALPHAS:
                    mixed = blend_scores(svd_scores, scores[c], a)
                    results[f"Hybrid k={k} content={c} alpha={a}"] = evaluate_scores(
                        mixed, R, relevant, cold_relevant, cold_mask, len(movie_ids)
                    )
        return results

    svd_name = f"PureSVD (k={svd_k})"
    scores[svd_name] = PureSVD(svd_k).fit(R).score_users(R)
    scores[f"Hybrid (PureSVD + {content}, alpha={alpha})"] = blend_scores(scores[svd_name], scores[content], alpha)
    for name, s in scores.items():
        results[name] = evaluate_scores(s, R, relevant, cold_relevant, cold_mask, len(movie_ids))
        print(f"{name:55s} ndcg@{K}={results[name][f'ndcg@{K}']:.4f}")
    return results


def plot(metrics: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    names = metrics["model"]
    axes[0].barh(names, metrics[f"ndcg@{K}"], color="#4c72b0")
    axes[0].set_title(f"NDCG@{K} (all test movies)")
    axes[1].barh(names, metrics[f"recall@{K} (cold-start)"], color="#dd8452")
    axes[1].set_title(f"Recall@{K} on cold-start movies (no train ratings)")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    movies, ratings = load_data(os.path.join(DATA_DIR, "movies.csv"), os.path.join(DATA_DIR, "ratings.csv"))
    emb = load_or_build_embeddings(movies, os.path.join(DATA_DIR, "embeddings.npz"))

    content_models = {
        "TF-IDF (genres only)": ContentRecommender(text_cols=["genres"]).fit(movies),
        "TF-IDF (genres + overview)": ContentRecommender(text_cols=["genres", "overview"]).fit(movies),
        "Embeddings": ContentRecommender().fit(movies, vectors=emb),
        "Embeddings + genres": ContentRecommender().fit(movies, vectors=with_genres(emb, movies)),
    }

    train, test = temporal_split(ratings, test_frac=0.2)

    # pick k and alpha on a validation split taken from the train set only
    train_inner, val = temporal_split(train, test_frac=0.2)
    tuning = pd.DataFrame(run(train_inner, val, movies, content_models)).T
    tuning.index.name = "setting"
    tuning.to_csv(os.path.join(RESULTS_DIR, "tuning.csv"), float_format="%.4f")

    svd_rows = tuning[tuning.index.str.startswith("PureSVD")]
    best_k = int(svd_rows[f"ndcg@{K}"].idxmax().split("=")[1])
    hyb_rows = tuning[tuning.index.str.startswith(f"Hybrid k={best_k} ")]
    best = hyb_rows[f"ndcg@{K}"].idxmax()
    best_content = best.split("content=")[1].split(" alpha=")[0]
    best_alpha = float(best.split("alpha=")[1])
    print(f"Best on validation: k={best_k}, content={best_content}, alpha={best_alpha}")

    results = run(train, test, movies, content_models, svd_k=best_k, content=best_content, alpha=best_alpha)
    metrics = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    metrics["users"] = metrics["users"].astype(int)
    metrics.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False, float_format="%.4f")
    plot(metrics, os.path.join(RESULTS_DIR, "metrics.png"))
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
