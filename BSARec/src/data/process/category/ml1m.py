from collections import Counter
import json
import pandas as pd

from _utils import DATA_DIR, RAW_DATA_DIR


CATEGORY_MAPPING_DIR = DATA_DIR / "category_maps" / "ml-1m"


def _load_movies() -> pd.DataFrame:
    movies_data_path = RAW_DATA_DIR / "ml-1m" / "movies.dat"
    # Load the movies data
    movies_df = pd.read_csv(
        movies_data_path,
        delimiter="::",
        encoding="latin-1",
        header=None,
        names=["movie_id", "movie_name", "movie_categories"],
        engine="python",
    )
    # Transform the movies category from string to list
    # e.g. "Thriller|Action" -> ["Thriller", "Action"]
    movies_df["movie_categories"] = movies_df["movie_categories"].apply(
        lambda x: x.split("|")
    )
    return movies_df


def create_movie_category_mapping():
    movie_df = _load_movies()

    # Only keep the mapping for artistID <-> most_common_category
    mapping = movie_df[["movie_id", "movie_categories"]]

    mapping = {row.movie_id: row.movie_categories for row in mapping.itertuples()}

    # Save the mapping as json
    CATEGORY_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = CATEGORY_MAPPING_DIR / "movie_category_mapping.json"
    with mapping_file.open("w", newline="", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4)

    print(f"Artist category mapping stored in `{mapping_file}`")
