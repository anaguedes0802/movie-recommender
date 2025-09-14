"""
Content-based model using cosine similarity between movie vectors.

By default the vectors are TF-IDF of 'genres' and 'overview' joined into a
single text field. Precomputed vectors (e.g. sentence embeddings) can be
passed to fit() instead.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(
        self,
        text_cols: Optional[List[str]] = None,
        max_features: int = 50_000,
        min_df: int = 2,
    ) -> None:
        self.text_cols = text_cols or ["genres", "overview"]
        # english stop words because TMDB overviews are usually in English
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            stop_words="english",
        )
        self._vectors = None
        self._index: Optional[pd.Index] = None
        self.movies: Optional[pd.DataFrame] = None

    def _combine_text(self, df: pd.DataFrame) -> pd.Series:
        def norm(v) -> str:
            if pd.isna(v):
                return ""
            # MovieLens genres look like "Adventure|Comedy|..."
            return str(v).replace("|", " ").strip()

        blobs = []
        for _, row in df.iterrows():
            parts = [norm(row.get(col, "")) for col in self.text_cols]
            blobs.append(" ".join(p for p in parts if p))
        return pd.Series(blobs, index=df.index)

    def fit(self, movies: pd.DataFrame, vectors=None) -> "ContentRecommender":
        if "movieId" not in movies.columns or "title" not in movies.columns:
            raise ValueError("movies needs 'movieId' and 'title' columns")
        self.movies = movies.copy()
        self._index = pd.Index(self.movies["movieId"].values)
        if vectors is not None:
            if vectors.shape[0] != len(self.movies):
                raise ValueError("vectors must have one row per movie")
            self._vectors = vectors
        else:
            corpus = self._combine_text(self.movies)
            self._vectors = self.vectorizer.fit_transform(corpus)
        return self

    def _assert_fitted(self) -> None:
        if self._vectors is None or self._index is None or self.movies is None:
            raise RuntimeError("Model is not fitted yet (call .fit).")

    def recommend_by_title(self, title_query: str, top_k: int = 10) -> pd.DataFrame:
        """Finds titles that contain the query (case-insensitive) and recommends similar ones."""
        self._assert_fitted()
        # MovieLens writes titles like "Godfather, The (1972)", so also search "The Godfather (1972)"
        readable = self.movies["title"].str.replace(r"^(.*), (The|A|An) \((\d{4})\)$", r"\2 \1 (\3)", regex=True)
        mask = (
            self.movies["title"].str.contains(title_query, case=False, na=False, regex=False)
            | readable.str.contains(title_query, case=False, na=False, regex=False)
        )
        if not mask.any():
            return pd.DataFrame(columns=["movieId", "title", "score"])
        seed_ids = self.movies.loc[mask, "movieId"].tolist()
        return self.recommend_for_items(seed_ids, top_k=top_k)

    def recommend_for_items(self, movie_ids: List[int], top_k: int = 10) -> pd.DataFrame:
        """Averages the seed vectors and ranks every movie by similarity. Seeds are excluded."""
        self._assert_fitted()
        idxs = []
        for mid in movie_ids:
            try:
                idxs.append(self._index.get_loc(mid))
            except KeyError:
                continue

        if not idxs:
            return pd.DataFrame(columns=["movieId", "title", "score"])

        # mean of a sparse matrix gives np.matrix, so convert to a plain array
        seed_vec = np.asarray(self._vectors[idxs].mean(axis=0)).reshape(1, -1)
        sims = cosine_similarity(seed_vec, self._vectors).ravel()

        # don't recommend the seeds themselves
        for i in idxs:
            sims[i] = -1.0

        return self._top(sims, top_k)

    def recommend_for_vector(self, vec, top_k: int = 10) -> pd.DataFrame:
        """Ranks movies by similarity to any vector in the same space (e.g. an embedded text query)."""
        self._assert_fitted()
        sims = cosine_similarity(np.asarray(vec).reshape(1, -1), self._vectors).ravel()
        return self._top(sims, top_k)

    def score_users(self, weights) -> np.ndarray:
        """
        Scores every movie for many users at once (used in the evaluation).
        weights is a users x movies matrix, each user profile is the weighted
        sum of the vectors of the movies they liked.
        """
        self._assert_fitted()
        profiles = weights @ self._vectors
        return cosine_similarity(profiles, self._vectors)

    def _top(self, sims: np.ndarray, top_k: int) -> pd.DataFrame:
        take = max(top_k, 1)
        top = np.argsort(-sims)[:take]
        cols = ["movieId", "title", "overview"] if "overview" in self.movies.columns else ["movieId", "title"]
        out = self.movies.iloc[top][cols].copy()
        out["score"] = sims[top]
        return out.reset_index(drop=True)

    def similarity_matrix(self, movie_ids: List[int]) -> pd.DataFrame:
        """Cosine similarity matrix between the given movies (used for the heatmap)."""
        self._assert_fitted()
        idxs = []
        titles = []
        for mid in movie_ids:
            try:
                i = self._index.get_loc(mid)
                idxs.append(i)
                titles.append(self.movies.iloc[i]["title"])
            except KeyError:
                continue

        if len(idxs) < 2:
            return pd.DataFrame()

        sub = self._vectors[idxs]
        mat = cosine_similarity(sub)
        return pd.DataFrame(mat, index=titles, columns=titles)
