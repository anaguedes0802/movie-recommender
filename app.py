"""
Streamlit app for the movie recommender.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.prepare_data import load_data
from src.content_model import ContentRecommender
from src.collaborative import PureSVD
from src.embeddings import load_or_build_embeddings, load_model, with_genres
from src.evaluation import ratings_matrix, liked_weights
from src.hybrid_recommender import hybrid_rank, blend_scores
from src.utils import titles_to_ids

st.set_page_config(page_title="Movie Recommender", layout="wide")

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npz")

# picked on the validation split, see evaluate.py and results/tuning.csv
SVD_K = 16
ALPHA = 0.7


@st.cache_data(show_spinner=False)
def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_data(MOVIES_PATH, RATINGS_PATH)


@st.cache_resource(show_spinner=False)
def _fit_tfidf(movies: pd.DataFrame) -> ContentRecommender:
    return ContentRecommender(text_cols=["genres", "overview"]).fit(movies)


@st.cache_resource(show_spinner="Computing embeddings (only the first time, takes a few minutes)...")
def _fit_embeddings(movies: pd.DataFrame):
    """Returns (embeddings model, embeddings + genres model), or (None, None) if it fails."""
    try:
        emb = load_or_build_embeddings(movies, EMB_PATH)
    except Exception as e:
        log.warning("Could not load the embeddings: %s", e)
        return None, None
    plain = ContentRecommender().fit(movies, vectors=emb)
    with_gen = ContentRecommender().fit(movies, vectors=with_genres(emb, movies))
    return plain, with_gen


@st.cache_resource(show_spinner=False)
def _text_model():
    return load_model()


@st.cache_resource(show_spinner=False)
def _fit_svd(ratings: pd.DataFrame, movies: pd.DataFrame):
    user_ids = np.sort(ratings["userId"].unique())
    R = ratings_matrix(ratings, user_ids, movies["movieId"].values)
    return PureSVD(SVD_K).fit(R), R, user_ids


def show_table(df: pd.DataFrame, extra_cols: list[str] | None = None):
    cols = ["title"] + (["overview"] if "overview" in df.columns else []) + ["score"] + (extra_cols or [])
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def heatmap(model: ContentRecommender, movie_ids: list[int]):
    sim_df = model.similarity_matrix(movie_ids)
    if sim_df.empty:
        st.info("You need at least 2 movies to draw the heatmap.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_df.values)
    ax.set_xticks(range(sim_df.shape[1]))
    ax.set_yticks(range(sim_df.shape[0]))
    ax.set_xticklabels(sim_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sim_df.index, fontsize=8)
    ax.set_title("Cosine similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def similar_movies_tab(movies, ratings, model, top_k, use_quality, quality_alpha):
    title_query = st.text_input("Search by title (contains):", "Toy Story")
    seed_titles = st.multiselect(
        "...or pick a few favourites (overrides the search).",
        options=movies["title"].tolist(),
    )
    seed_ids = titles_to_ids(movies, seed_titles)

    # get more candidates than needed so the quality ranking has something to reorder
    if seed_ids:
        candidates = model.recommend_for_items(seed_ids, top_k=top_k * 3)
    else:
        candidates = model.recommend_by_title(title_query, top_k=top_k * 3)

    if candidates.empty:
        st.info("No candidates found. Try another title or pick some movies.")
        return

    if use_quality:
        ranked = hybrid_rank(candidates, ratings, alpha=quality_alpha).head(top_k)
        show_table(ranked, ["hybrid"])
    else:
        ranked = candidates.head(top_k)
        show_table(ranked)

    with st.expander("Similarity heatmap"):
        ids = seed_ids if len(seed_ids) >= 2 else ranked["movieId"].tolist()[:10]
        heatmap(model, ids)


def describe_tab(emb_model, top_k):
    if emb_model is None:
        st.warning("This needs the sentence embeddings (install sentence-transformers).")
        return
    query = st.text_input("Describe the movie you want to watch:", "a heist that goes wrong in a small town")
    if not query.strip():
        return
    vec = _text_model().encode([query], normalize_embeddings=True)[0]
    show_table(emb_model.recommend_for_vector(vec, top_k=top_k))


def user_tab(movies, ratings, content_model, top_k):
    svd, R, user_ids = _fit_svd(ratings, movies)
    user_id = st.selectbox("MovieLens user", user_ids)
    u = int(np.searchsorted(user_ids, user_id))

    user_ratings = ratings[ratings["userId"] == user_id].merge(movies[["movieId", "title"]], on="movieId")
    st.write(f"User {user_id} rated {len(user_ratings)} movies. Favourites:")
    st.dataframe(
        user_ratings.sort_values(["rating", "timestamp"], ascending=False)[["title", "rating"]].head(8),
        use_container_width=True, hide_index=True,
    )

    row = R[u]
    cf = svd.score_users(row)
    content = content_model.score_users(liked_weights(row))
    scores = blend_scores(cf, content, ALPHA)[0]
    scores[row.nonzero()[1]] = -np.inf  # don't recommend what they already rated

    top = np.argsort(-scores)[:top_k]
    recs = movies.iloc[top][["movieId", "title", "overview"]].copy()
    recs["score"] = scores[top]
    st.write(f"Recommendations (PureSVD k={SVD_K} + content, alpha={ALPHA}):")
    show_table(recs)


def evaluation_tab():
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    if not os.path.exists(metrics_path):
        st.info("No results yet. Run `python evaluate.py` first.")
        return
    st.write(
        "Temporal split (last 20% of each user's ratings in test), relevant = rating >= 4. "
        "Cold-start = ranking only movies that have no ratings in train."
    )
    st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
    png = os.path.join(RESULTS_DIR, "metrics.png")
    if os.path.exists(png):
        st.image(png)


def main():
    st.title("Movie Recommender System")
    st.caption("Content-based (TF-IDF / sentence embeddings) + collaborative filtering. Dataset: MovieLens small.")

    if not os.path.exists(MOVIES_PATH) or not os.path.exists(RATINGS_PATH):
        st.error("Missing files in `data/`. Run `python download_data.py` first.")
        st.stop()

    movies, ratings = _load()
    tfidf = _fit_tfidf(movies)
    emb_model, emb_genres_model = _fit_embeddings(movies)

    st.sidebar.header("Settings")
    options = ["TF-IDF (genres + overview)"]
    if emb_genres_model is not None:
        options.insert(0, "Embeddings + genres")
    choice = st.sidebar.selectbox("Content model", options)
    content_model = emb_genres_model if choice == "Embeddings + genres" else tfidf

    top_k = st.sidebar.slider("Number of recommendations", 5, 30, 10, step=1)
    use_quality = st.sidebar.checkbox("Boost well-rated movies (Similar movies tab)", value=False)
    quality_alpha = st.sidebar.slider("Content weight (alpha)", 0.0, 1.0, 0.8, step=0.05)

    tab1, tab2, tab3, tab4 = st.tabs(["Similar movies", "Describe a movie", "For a user", "Evaluation"])
    with tab1:
        similar_movies_tab(movies, ratings, content_model, top_k, use_quality, quality_alpha)
    with tab2:
        describe_tab(emb_model, top_k)
    with tab3:
        user_tab(movies, ratings, content_model, top_k)
    with tab4:
        evaluation_tab()


if __name__ == "__main__":
    main()
