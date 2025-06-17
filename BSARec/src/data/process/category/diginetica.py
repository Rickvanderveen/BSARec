import json
import pandas as pd

from _utils import DATA_DIR, RAW_DATA_DIR
from category import DatasetCategoryMapCreator


CATEGORY_MAPPING_DIR = DATA_DIR / "category_maps" / "Diginetica"


def _load_categories() -> pd.DataFrame:
    categories_path = RAW_DATA_DIR / "Diginetica" / "product_categories.csv"
    # Load the movies data
    categories_df = pd.read_csv(
        categories_path,
        delimiter=";",
        encoding="utf-8",
    )

    return categories_df


def _load_user_item_data() -> pd.DataFrame:
    data_path = RAW_DATA_DIR / "Diginetica" / "diginetica_train.csv"
    df = pd.read_csv(data_path, delimiter=";", header=0, names=["userId", "itemId", "timeframe", "eventdate"])

    # Drop all users which are nan
    df = df.dropna(subset=["userId", "itemId", "timeframe", "eventdate"])
    
    return df


def create_product_category_mapping():
    category_df = _load_categories()

    mapping = {row.itemId: row.categoryId for row in category_df.itertuples()}

    # Save the mapping as json
    CATEGORY_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = CATEGORY_MAPPING_DIR / "product_category_mapping.json"
    with mapping_file.open("w", newline="", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4)

    print(f"Product category mapping stored in `{mapping_file}`")


def create_popularity_map(top_percentage_popularity: float = 0.2):
    user_item_df = _load_user_item_data()
    count = (
        user_item_df["itemId"]
        .value_counts(sort=True, ascending=False)
        .reset_index()
    )

    top_percent_count = int(len(count) * top_percentage_popularity)

    mapping = {}
    for idx, row in enumerate(count.itertuples()):
        popularity_category = "unpop"
        if idx <= top_percent_count:
            popularity_category = "pop"

        item_id = row.itemId
        mapping[item_id] = popularity_category
    
    # Save the mapping as json
    CATEGORY_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = CATEGORY_MAPPING_DIR / "product_popularity_mapping.json"
    with mapping_file.open("w", newline="", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4)
    
    print(f"Product popularity mapping stored in `{mapping_file}`")



class DigineticaCategoryCreator(DatasetCategoryMapCreator):
    @staticmethod
    def create_native_map():
        create_product_category_mapping()
    
    @staticmethod
    def create_popularity_map():
        create_popularity_map()