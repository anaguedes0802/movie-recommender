# Downloads MovieLens small (GroupLens) and the plot overviews from
# "The Movies Dataset" on Kaggle, and saves everything in data/
import io
import os
import zipfile

import kagglehub
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens():
    r = requests.get(MOVIELENS_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))
        ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
        links = pd.read_csv(z.open("ml-latest-small/links.csv"))
    return movies, ratings, links


def download_overviews():
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset", path="movies_metadata.csv")
    # on Windows kagglehub sometimes leaves the file zipped
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            meta = pd.read_csv(z.open(z.namelist()[0]), usecols=["id", "overview"], low_memory=False)
    else:
        meta = pd.read_csv(path, usecols=["id", "overview"], low_memory=False)
    # a few rows have broken ids, drop them
    meta["id"] = pd.to_numeric(meta["id"], errors="coerce")
    meta = meta.dropna(subset=["id"]).drop_duplicates("id")
    meta["tmdbId"] = meta["id"].astype("Int64")
    return meta[["tmdbId", "overview"]]


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    movies, ratings, links = download_movielens()
    print("Movies:", movies.shape, "| Ratings:", ratings.shape)

    try:
        overviews = download_overviews()
        links["tmdbId"] = links["tmdbId"].astype("Int64")
        movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
        movies = movies.merge(overviews, on="tmdbId", how="left").drop(columns="tmdbId")
    except Exception as e:
        print("Could not download the overviews, the model will only use genres:", e)
        movies["overview"] = ""

    movies["overview"] = movies["overview"].fillna("").str.strip()
    n = (movies["overview"] != "").sum()
    print(f"Overviews found for {n} of {len(movies)} movies")

    movies.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False, encoding="utf-8")
    ratings.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False, encoding="utf-8")
    print("Saved to:", DATA_DIR)
