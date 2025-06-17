import argparse

from category import ml1m
from category.lastfm import LastFMCategoryCreator
from category.diginetica import DigineticaCategoryCreator


CATEGORY_DATASET_CHOICES = ["Diginetica", "LastFM", "ML-1M"]
CATEGORY_TYPES = ["Popularity", "Native"]


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

    parser.add_argument(
        "--category",
        "-c",
        choices=CATEGORY_TYPES,
        default="Native",
        help=(
        "Category map type to create. "
        "Options:\n"
        f"  - {CATEGORY_TYPES[0]}: Use dataset-specific categories\n"
        f"  - {CATEGORY_TYPES[1]}: Use a popularity-based category set"
    ),
    )

    # Parse and return arguments
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        print("Create Diginetica product-category mapping")
        DigineticaCategoryCreator.create(args.category)
        print("Create LastFM artist-genre mapping")
        LastFMCategoryCreator.create(args.category)
        print("Create ML-1M movie-category mapping")
        ml1m.create_movie_category_mapping()

        return

    print(f"Creating category map for dataset: {args.dataset}")
    dataset = args.dataset.lower()
    if dataset == "diginetica":
        DigineticaCategoryCreator.create(args.category)
    elif dataset == "lastfm":
        LastFMCategoryCreator.create(args.category)
    elif dataset == "ml-1m":
        ml1m.create_movie_category_mapping()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
