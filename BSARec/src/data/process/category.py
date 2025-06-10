import argparse

from category import lastfm, ml1m


CATEGORY_DATASET_CHOICES = ["LastFM", "ML-1M"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create category map for a dataset.")

    group = parser.add_mutually_exclusive_group(required=True)

    # Add a required choice argument
    group.add_argument(
        "--dataset",
        choices=CATEGORY_DATASET_CHOICES,
        help=f"Dataset to process. Choices: {', '.join(CATEGORY_DATASET_CHOICES)}",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Process all available datasets",
    )

    # Parse and return arguments
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        print("Create LastFM artist-genre mapping")
        lastfm.create_artist_genre_mapping()
        print("Create ML-1M movie-category mapping")
        ml1m.create_movie_category_mapping()

        return

    print(f"Creating category map for dataset: {args.dataset}")
    dataset = args.dataset.lower()
    if dataset == "lastfm":
        lastfm.create_artist_genre_mapping()
    elif dataset == "ml-1m":
        ml1m.create_movie_category_mapping()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
