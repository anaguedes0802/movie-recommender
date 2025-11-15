# Movie Recommender

Movie recommender on MovieLens that compares content-based models (TF-IDF and sentence embeddings of the plot) with collaborative filtering, plus a hybrid of both, evaluated offline. Includes a Streamlit app.

Personal project. First version in September 2025 (TF-IDF on genres + Streamlit app), extended in October and November 2025 with plot overviews, sentence embeddings, collaborative filtering and the offline evaluation.

![Similar movies tab](docs/similar.png)

<p>
  <img src="docs/describe.png" alt="Describe a movie tab" width="49%">
  <img src="docs/user.png" alt="For a user tab" width="49%">
</p>

## Problem

Two questions:

1. Given a few movies someone likes (or a text description), which other movies are similar?
2. Given a user's rating history, which movies should we recommend, including movies that nobody has rated yet (cold start)?

The first version only compared genres. Many movies have exactly the same genres, so the results were mostly ties (for "Toy Story" the whole top 10 had the same score). It also had no evaluation, so there was no way to tell if it was any good.

## Data

- **MovieLens latest-small** from GroupLens: 9,742 movies, 100,836 ratings, 610 users.
- **Plot overviews** from "The Movies Dataset" on Kaggle (`rounakbanik/the-movies-dataset`), joined through the TMDB ids in MovieLens `links.csv`. 9,525 of the 9,742 movies (97.8%) got an overview.

Neither is included in the repo. `download_data.py` downloads both and writes `data/movies.csv` (with an `overview` column) and `data/ratings.csv`.

- MovieLens page: https://grouplens.org/datasets/movielens/latest/
- The Movies Dataset: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

## Approach

**Content-based** (`src/content_model.py`). Every movie becomes a vector and similar movies are found with cosine similarity:

- TF-IDF of the genres only (the original version)
- TF-IDF of genres + overview
- Sentence embeddings of "title. genres. overview" with `all-MiniLM-L6-v2` (`src/embeddings.py`)
- The same embeddings with a one-hot genre vector appended ("Embeddings + genres")

For a user, the profile is the weighted sum of the vectors of the movies they rated 4 or more.

**Collaborative filtering** (`src/collaborative.py`):

- Popularity (baseline)
- ItemKNN: item-item cosine similarity on the rating columns
- PureSVD (Cremonesi et al., 2010): truncated SVD of the ratings matrix

**Hybrid** (`src/hybrid_recommender.py`): each model's scores are min-max scaled per user and mixed as `alpha * PureSVD + (1 - alpha) * content`.

**Evaluation** (`src/evaluation.py`, `evaluate.py`):

- Temporal split per user: the most recent 20% of each user's ratings go to test.
- A test movie is relevant if it was rated 4 or more. Movies already rated in train are never recommended.
- Metrics: precision, recall and NDCG at 10, plus catalogue coverage (share of movies recommended to at least one user).
- Cold-start metric: recall@10 when ranking only the 1,503 movies that have no ratings in train.
- The PureSVD factors (8 to 128), the content model used in the hybrid and `alpha` were chosen on a validation split taken from the train set (the last 20% of each user's train ratings). The test set was only used for the final numbers. All settings tried are in `results/tuning.csv`.

## Results

Test set, 592 users with at least one relevant movie (from `results/metrics.csv`):

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Recall@10 cold-start |
|---|---|---|---|---|---|
| Popularity | 0.0571 | 0.0497 | 0.0743 | 0.0101 | 0.0094 |
| TF-IDF (genres only) | 0.0064 | 0.0083 | 0.0102 | 0.2613 | 0.0205 |
| TF-IDF (genres + overview) | 0.0044 | 0.0038 | 0.0067 | 0.0497 | 0.0299 |
| Embeddings | 0.0039 | 0.0059 | 0.0050 | 0.0665 | 0.0226 |
| Embeddings + genres | 0.0073 | 0.0072 | 0.0091 | 0.1018 | 0.0332 |
| ItemKNN | 0.0713 | 0.0743 | 0.0902 | 0.0491 | 0.0094 |
| PureSVD (k=16) | 0.0856 | 0.0851 | **0.1085** | 0.0457 | 0.0094 |
| Hybrid (PureSVD + Embeddings + genres, alpha=0.7) | 0.0846 | 0.0849 | 0.1082 | 0.0521 | **0.0332** |

`python evaluate.py` also draws these two metrics as a bar chart in `results/metrics.png`.

What this shows:

- **Collaborative filtering wins clearly on accuracy.** PureSVD has the best NDCG@10. Every content model is below even the popularity baseline. On this dataset, what other users rated says much more than the plot or the genres.
- **Content models are the only ones that work in cold start.** For movies with no ratings in train, PureSVD, ItemKNN and popularity cannot tell them apart (0.0094 is what you get from an arbitrary order). The content models find the liked ones up to 3.5x more often, and "Embeddings + genres" is the best of them.
- **The hybrid gets both.** It keeps PureSVD's accuracy (0.1082 vs 0.1085) and the cold-start recall of the best content model (0.0332). On the validation split it also improved NDCG@10 (0.1265 vs 0.1181 for PureSVD alone), but that gain did not carry over to the test set.
- **The plot text helps only in cold start.** On NDCG@10 all the text-based models are below TF-IDF on genres alone. In cold start they are better (0.0226 to 0.0332 vs 0.0205), and appending the genres to the embeddings gives the best of both.

Examples from the app (Embeddings + genres):

- *The Godfather* → The Godfather: Part II, Donnie Brasco, The Godfather: Part III, Rocco and His Brothers, The Funeral, Scarface (1932), Boy A, Goodfellas
- *Toy Story* (all three) → The Lego Movie, Monsters, Inc., The Adventures of Rocky and Bullwinkle, Charlie and the Chocolate Factory, The Peanuts Movie
- Text search "a heist that goes wrong in a small town" → American Heist, Tower Heist, The Maiden Heist, Heist (2001), The Good Thief

Limitations:

- Small dataset (610 users) and a single split. There are no confidence intervals, so small differences (like hybrid vs PureSVD) are not meaningful.
- The cold-start result is based on 120 users and 619 user-movie pairs.
- The overviews come from a 2017 dataset, and 217 movies have none.

## App

`streamlit run app.py` has four tabs:

- **Similar movies**: search a title or pick favourites, with an option to boost well-rated movies (Bayesian average of the ratings) and a similarity heatmap.
- **Describe a movie**: free-text search with the sentence embeddings.
- **For a user**: pick a MovieLens user, see their favourites and the hybrid recommendations.
- **Evaluation**: the table and chart above.

Screenshots of the first three tabs are at the top of this README.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

python download_data.py       # MovieLens + overviews into data/
python evaluate.py            # optional, rewrites results/ (a few minutes)
streamlit run app.py
```

The embeddings are computed the first time (a few minutes on CPU) and saved to `data/embeddings.npz`.

Tests: `pip install pytest` and then `pytest tests`.

## Project structure

```
├── app.py                    # Streamlit app
├── download_data.py          # MovieLens + overviews into data/
├── evaluate.py               # offline evaluation, writes results/
├── requirements.txt
├── docs/                     # app screenshots
├── results/                  # metrics.csv, tuning.csv (+ metrics.png when you run evaluate.py)
├── src/
│   ├── collaborative.py      # Popularity, ItemKNN, PureSVD
│   ├── content_model.py      # cosine similarity on TF-IDF or embeddings
│   ├── embeddings.py         # sentence embeddings (+ genres)
│   ├── evaluation.py         # split, metrics
│   ├── hybrid_recommender.py # score blending + Bayesian average ranking
│   ├── prepare_data.py       # loads and checks the CSVs
│   └── utils.py
└── tests/
    └── test_evaluation.py
```

## Possible next steps

Not done:

- Repeat the evaluation with several splits (or the bigger MovieLens 25M) to get confidence intervals.
- Try a bigger embedding model and learning-to-rank instead of a fixed `alpha`.
