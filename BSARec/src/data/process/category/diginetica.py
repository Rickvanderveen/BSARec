import json
import pandas as pd

from _utils import DATA_DIR, RAW_DATA_DIR


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


def create_product_category_mapping():
    category_df = _load_categories()

    mapping = {row.itemId: row.categoryId for row in category_df.itertuples()}

    # Save the mapping as json
    CATEGORY_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = CATEGORY_MAPPING_DIR / "product_category_mapping.json"
    with mapping_file.open("w", newline="", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4)

    print(f"Product category mapping stored in `{mapping_file}`")
