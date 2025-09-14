from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

log = logging.getLogger(__name__)


def _read_csv_smart(path: Path) -> pd.DataFrame:
    """Tries UTF-8 first and falls back to common Windows encodings."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path} with any of {encodings}")


def load_data(movies_path: str | Path, ratings_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies_path = Path(movies_path)
    ratings_path = Path(ratings_path)

    if not movies_path.exists():
        raise FileNotFoundError(f"File not found: {movies_path.resolve()}")
    if not ratings_path.exists():
        raise FileNotFoundError(f"File not found: {ratings_path.resolve()}")

    log.info("Loading movies from %s", movies_path)
    movies = _read_csv_smart(movies_path)
    log.info("Loading ratings from %s", ratings_path)
    ratings = _read_csv_smart(ratings_path)

    required_movies = {"movieId", "title"}
    if not required_movies.issubset(movies.columns):
        raise ValueError(f"movies.csv needs the columns: {required_movies}")

    for col in ("genres", "overview"):
        if col not in movies.columns:
            log.warning("Optional column '%s' missing in movies.csv, creating it empty.", col)
            movies[col] = ""

    movies = movies.drop_duplicates(subset="movieId").reset_index(drop=True)
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    for c in ("userId", "movieId"):
        if ratings[c].dtype.kind not in "iu":
            log.warning("Converting '%s' to integer.", c)
            ratings[c] = pd.to_numeric(ratings[c], errors="coerce").astype("Int64")

    return movies, ratings
