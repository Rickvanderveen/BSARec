from collections import Counter
import json

import pandas as pd
import difflib

from _utils import DATA_DIR, RAW_DATA_DIR


CATEGORY_MAPPING_DIR = DATA_DIR / "category_maps" / "LastFM"


STANDARD_GENRE_DICT = {
    "Rock": [
        "rock",
        "classic rock",
        "alternative rock",
        "indie rock",
        "hard rock",
        "soft rock",
        "psychedelic rock",
        "garage rock",
        "punk rock",
        "folk rock",
        "glam rock",
        "southern rock",
        "arena rock",
        "progressive rock",
        "art rock",
        "grunge",
    ],
    "Pop": [
        "pop",
        "dance pop",
        "electropop",
        "teen pop",
        "pop rock",
        "synthpop",
        "indie pop",
        "dream pop",
        "k-pop",
        "j-pop",
        "bubblegum pop",
        "power pop",
    ],
    "Hip-Hop": [
        "hip-hop",
        "hip hop",
        "rap",
        "trap",
        "gangsta rap",
        "conscious hip hop",
        "alternative hip hop",
        "old school hip hop",
        "boom bap",
        "east coast rap",
        "west coast rap",
        "dirty south",
        "drill",
        "mumble rap",
    ],
    "Electronic": [
        "electronic",
        "edm",
        "house",
        "techno",
        "trance",
        "drum and bass",
        "dnb",
        "dubstep",
        "electro",
        "idm",
        "ambient",
        "glitch",
        "breakbeat",
        "future bass",
        "synthwave",
        "chillwave",
        "deep house",
        "progressive house",
        "hardstyle",
    ],
    "Jazz": [
        "jazz",
        "smooth jazz",
        "bebop",
        "cool jazz",
        "hard bop",
        "free jazz",
        "fusion",
        "latin jazz",
        "swing",
        "jazz funk",
        "vocal jazz",
        "avant-garde jazz",
    ],
    "Classical": [
        "classical",
        "baroque",
        "romantic",
        "modern classical",
        "orchestral",
        "chamber music",
        "opera",
        "symphony",
        "piano",
        "early music",
        "contemporary classical",
        "minimalism",
    ],
    "Metal": [
        "metal",
        "heavy metal",
        "black metal",
        "death metal",
        "thrash metal",
        "doom metal",
        "power metal",
        "symphonic metal",
        "folk metal",
        "nu metal",
        "progressive metal",
        "metalcore",
        "grindcore",
        "industrial metal",
    ],
    "R&B": [
        "r&b",
        "rnb",
        "soul",
        "neo soul",
        "contemporary r&b",
        "quiet storm",
        "motown",
        "funk",
        "new jack swing",
        "blue-eyed soul",
    ],
    "Reggae": [
        "reggae",
        "roots reggae",
        "dub",
        "dancehall",
        "ska",
        "rocksteady",
        "ragga",
    ],
    "Country": [
        "country",
        "alt-country",
        "country rock",
        "bluegrass",
        "americana",
        "honky tonk",
        "country pop",
        "traditional country",
        "outlaw country",
        "neo-traditional country",
    ],
    "Blues": [
        "blues",
        "delta blues",
        "electric blues",
        "chicago blues",
        "country blues",
        "blues rock",
        "texas blues",
        "rhythm and blues",
    ],
    "Folk": [
        "folk",
        "indie folk",
        "folk rock",
        "contemporary folk",
        "traditional folk",
        "acoustic",
        "singer-songwriter",
        "americana",
        "celtic",
        "bluegrass",
    ],
    "Latin": [
        "latin",
        "reggaeton",
        "latin pop",
        "latin rock",
        "salsa",
        "bachata",
        "merengue",
        "cumbia",
        "tango",
        "bossa nova",
        "mariachi",
        "norteño",
        "latin trap",
    ],
    "World": [
        "world",
        "world music",
        "afrobeat",
        "afropop",
        "balkan",
        "celtic",
        "arabic",
        "flamenco",
        "fado",
        "indian",
        "klezmer",
        "traditional",
        "ethnic",
        "tribal",
    ],
    "Soundtrack": [
        "soundtrack",
        "ost",
        "original soundtrack",
        "score",
        "film score",
        "video game music",
        "musical",
        "broadway",
        "tv soundtrack",
        "anime soundtrack",
    ],
    "Experimental": [
        "experimental",
        "avant-garde",
        "noise",
        "glitch",
        "no wave",
        "industrial",
        "dark ambient",
        "sound art",
        "field recording",
        "drone",
        "electroacoustic",
    ],
    "Punk": [
        "punk",
        "punk rock",
        "pop punk",
        "hardcore punk",
        "post-punk",
        "emo",
        "skate punk",
        "anarcho-punk",
        "garage punk",
        "crust punk",
    ],
    "Religious": [
        "gospel",
        "christian",
        "christian rock",
        "worship",
        "praise",
        "contemporary christian",
        "ccm",
        "hymns",
        "sacred",
        "spiritual",
    ],
    "New Age": [
        "new age",
        "meditation",
        "relaxation",
        "healing",
        "ambient",
        "space music",
        "yoga music",
        "nature sounds",
    ],
    "Children": [
        "children",
        "kids",
        "nursery rhymes",
        "disney",
        "lullabies",
        "children's music",
    ],
}


def _map_tag_to_core_genre(user_tag, genre_dict, cutoff=0.8):
    """
    Maps a user-generated tag to the closest core genre from a standard genre dictionary.

    Parameters:
        user_tag (str): The input tag from a user (e.g., 'dream pop').
        genre_dict (dict): A dictionary mapping core genres to lists of tags.
        cutoff (float): Similarity threshold for fuzzy matching.

    Returns:
        str or None: The matched core genre, or None if no close match found.
    """
    user_tag = user_tag.strip().lower()

    # Flatten genre dictionary
    flat_tag_to_genre = {}
    for core_genre, tags in genre_dict.items():
        for tag in tags:
            flat_tag_to_genre[tag.lower()] = core_genre

    # Exact match
    if user_tag in flat_tag_to_genre:
        return flat_tag_to_genre[user_tag]

    # Fuzzy match
    all_tags = list(flat_tag_to_genre.keys())
    close_matches = difflib.get_close_matches(user_tag, all_tags, n=1, cutoff=cutoff)
    if close_matches:
        return flat_tag_to_genre[close_matches[0]]

    # Substring match
    for tag in all_tags:
        if user_tag in tag or tag in user_tag:
            return flat_tag_to_genre[tag]

    return "Other"  # No match found


def _load_tags() -> pd.DataFrame:
    tags_data_path = RAW_DATA_DIR / "LastFM" / "tags.dat"
    tag_df = pd.read_csv(
        tags_data_path,
        delimiter="\t",
        encoding="latin-1",
    )
    tag_df["category"] = tag_df["tagValue"].apply(
        lambda word: _map_tag_to_core_genre(word, STANDARD_GENRE_DICT)
    )

    return tag_df


def _load_user_tagged_artists() -> pd.DataFrame:
    path = RAW_DATA_DIR / "LastFM" / "user_taggedartists-timestamps.dat"
    user_tags_df = pd.read_csv(
        path,
        delimiter="\t",
        encoding="latin-1",
    )
    return user_tags_df


def _most_common_category(categories: list[str]) -> str:
    most_common_categories = Counter(categories).most_common(2)
    category = most_common_categories.pop(0)[0]
    if category == "Other" and most_common_categories:
        category = most_common_categories.pop(0)[0]
    return category


def _artists_category_mapping():
    tags_df = _load_tags()
    user_tags_df = _load_user_tagged_artists()

    user_artist_categories = pd.merge(tags_df, user_tags_df, on="tagID", how="inner")

    artist_genre_votes = user_artist_categories[["category", "artistID"]]
    artist_genre = (
        artist_genre_votes.groupby("artistID")["category"].apply(list).reset_index()
    )

    artist_genre["most_common_category"] = artist_genre["category"].apply(
        _most_common_category
    )
    return artist_genre


def create_artist_genre_mapping():
    # Create the mapping from the data files
    mapping_df = _artists_category_mapping()

    # Only keep the mapping for artistID <-> most_common_category
    mapping = mapping_df[["artistID", "most_common_category"]]

    mapping = {row.artistID: row.most_common_category for row in mapping.itertuples()}

    # Save the mapping as json
    CATEGORY_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = CATEGORY_MAPPING_DIR / "artist_category_mapping.json"
    with mapping_file.open("w", newline="", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4)

    print(f"Artist category mapping stored in `{mapping_file}`")
