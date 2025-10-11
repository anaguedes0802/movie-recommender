"""
Sentence embeddings for each movie (title + genres + overview).

The embeddings are saved to a .npz file so they are only computed once.
"""
from __future__ import annotations
import hashlib
import os
import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def movie_texts(movies: pd.DataFrame) -> list[str]:
    genres = movies["genres"].fillna("").str.replace("|", ", ", regex=False)
    overview = movies["overview"].fillna("") if "overview" in movies.columns else ""
    return (movies["title"] + ". " + genres + ". " + overview).tolist()


def load_or_build_embeddings(movies: pd.DataFrame, cache_path: str, model_name: str = MODEL_NAME) -> np.ndarray:
    texts = movie_texts(movies)
    # hash of the texts, so the cache is rebuilt if the data changes
    key = hashlib.md5((model_name + "\n".join(texts)).encode("utf-8")).hexdigest()

    if os.path.exists(cache_path):
        saved = np.load(cache_path)
        if "key" in saved and str(saved["key"]) == key:
            return saved["embeddings"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.savez(cache_path, embeddings=emb, key=key)
    return emb


def load_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def with_genres(emb: np.ndarray, movies: pd.DataFrame, weight: float = 1.0) -> np.ndarray:
    """
    Appends a one-hot genre vector to each embedding. The embeddings mostly
    follow the plot text, this pulls movies of the same genres closer together.
    """
    genres = movies["genres"].fillna("").str.get_dummies(sep="|").values.astype(float)
    norms = np.linalg.norm(genres, axis=1, keepdims=True)
    genres = genres / np.maximum(norms, 1e-12)
    return np.hstack([emb, weight * genres])
