from __future__ import annotations
from typing import Iterable
import pandas as pd


def titles_to_ids(movies: pd.DataFrame, titles: Iterable[str]) -> list[int]:
    """Converts a list of titles to movieIds, skipping titles that are not found."""
    m = movies.set_index("title")["movieId"]
    out = []
    for t in titles:
        try:
            out.append(int(m.loc[t]))
        except Exception:
            continue
    return out
