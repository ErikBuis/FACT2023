"""Script to preprocess the Visual Genome dataset as described in the paper."""

import json
import os
import pickle
from collections import defaultdict
from typing import Any, Tuple

from torchtext.vocab import GloVe


def load_samples(filename: str) -> list[dict[str, Any]]:
    """
    Load the samples from the original dataset.

    Args:
        filename: Path to the file containing object data.

    Returns:
        list[dict[str, Any]]: The original object samples.
    """
    print("[ LOADING SAMPLES ]".center(79, "-"))
    print(f"Loading samples from {filename}...", end=" ")
    with open(filename, "r") as f:
        samples = json.load(f)
    print("Done.")
    print()
    return samples


def save_samples_preprocessed(
        filename: str,
        samples_preprocessed: list[dict[str, list[dict[str, int]]]]):
    """
    Save the pre-processed samples to a file.

    Args:
        filename: Path to the file to save the pre-processed samples to.
        samples_preprocessed: The pre-processed samples.
    """
    print("[ SAVING SAMPLES ]".center(79, "-"))
    print(f"Saving pre-processed samples to {filename}...", end=" ")
    with open(filename, "w") as f:
        json.dump(samples_preprocessed, f)
    print("Done.")
    print()


def save_cat_mappings(filename: str, cat_mappings: dict[str, dict]):
    """
    Save the category mappings to a file.

    Args:
        filename: Path to the file to save the category mappings to.
        cat_mappings: The category mappings.
    """
    print("[ SAVING CATEGORY MAPPINGS ]".center(79, "-"))
    print(f"Saving category mappings to {filename}...", end=" ")
    with open(filename, "wb") as f:
        pickle.dump(cat_mappings, f)
    print("Done.")
    print()


def preprocess_vg(samples: list[dict[str, Any]], glove: GloVe) \
        -> Tuple[list[dict[str, list[dict[str, int]]]],
                 set[str]]:
    """
    Pre-process the data as described in the paper.

    Excerpt from the paper:
    "We only use images that have box-able annotations for our experiments.
    During pre-processing, we combined object categories based on their synset
    names, combined instances of the same category in the same image, and
    deleted categories that appear in less than 100 training images. In the
    end, there remains 106,215 out of 107,228 images, 1,208 out of 80,138
    object categories, and at most 47 object categories per image."

    Args:
        samples (list[dict[str, Any]]): The samples to pre-process.
        glove (GloVe): The GloVe word embeddings.

    Returns:
        Tuple:
            - list[dict[str, list[dict[str, int]]]]: The pre-processed samples.
            - set[str]: The pre-processed categories.
    """
    # Print some statistics before pre-processing.
    print("[ BEFORE PRE-PROCESSING ]".center(79, "-"))
    amount_images = len(samples)
    amount_instances = 0
    categories = set()
    for sample in samples:
        amount_instances += len(sample["objects"])
        for obj in sample["objects"]:
            for cat_token in obj["names"]:
                categories.add(cat_token)
    amount_categories = len(categories)
    print(f"{amount_images=}")
    print(f"{amount_instances=}")
    print(f"{amount_categories=}")
    print()

    # Count the number of instances in each pre-processed category.
    # Combine object categories based on their synset names.
    # Combine instances of the same category in the same image.
    print("[ PRE-PROCESSING ]".center(79, "-"))
    print("Pre-processing samples (pass 1/2)...", end=" ")
    cat_token_counts = defaultdict(int)
    samples_temp = []
    for sample in samples:
        objects = defaultdict(list)
        for obj in sample["objects"]:
            for synset in obj["synsets"]:
                cat_token = synset.partition(".")[0].partition("_")[0]
                if cat_token not in glove.stoi:
                    continue
                cat_token_counts[cat_token] += 1
                objects[cat_token].append({"x": obj["x"], "y": obj["y"],
                                           "h": obj["h"], "w": obj["w"]})
        if len(objects) > 0:
            samples_temp.append({"image_id": sample["image_id"],
                                 "objects": objects})
    print("Done.")

    # Delete categories that appear in less than 100 training images.
    print("Pre-processing samples (pass 2/2)...", end=" ")
    samples_preprocessed = []
    for sample in samples_temp:
        objects = {cat_token: bboxes
                   for cat_token, bboxes in sample["objects"].items()
                   if cat_token_counts[cat_token] >= 100}
        if len(objects) > 0:
            samples_preprocessed.append({"image_id": sample["image_id"],
                                         "objects": objects})
    print("Done.")
    print()

    # Print some statistics after preprocessing.
    print("[ AFTER PRE-PROCESSING ]".center(79, "-"))
    amount_images = len(samples_preprocessed)
    amount_instances = 0
    categories = set()
    for sample in samples_preprocessed:
        amount_instances += len(sample["objects"])
        for cat_token in sample["objects"]:
            categories.add(cat_token)
    amount_categories = len(categories)
    print(f"{amount_images=}")
    print(f"{amount_instances=}")
    print(f"{amount_categories=}")
    print()

    return samples_preprocessed, categories


def compute_cat_mappings(categories: set[str], glove: GloVe) \
        -> dict[str, dict]:
    """
    Compute mappings from COCO categories to GloVe indices.

    Args:
        ann_file: Path to the annotation file.

    Returns:
        dict: A dictionary containing the mappings. The keys are:
            - "stoi" (dict): COCO category to GloVe index.
            - "itos" (dict): GloVe index to COCO category.
    """
    print("[ COMPUTING CATEGORY MAPPINGS ]".center(79, "-"))
    print("Finding GloVe indices for each category...", end=" ")
    cat_mappings = {"stoi": {}, "itos": {}}
    for cat_token in categories:
        cat_mappings["stoi"][cat_token] = glove.stoi[cat_token]
        cat_mappings["itos"][glove.stoi[cat_token]] = cat_token
    print("Done.")
    print()
    return cat_mappings


def main():
    root = "./data/vg"
    glove = GloVe(name="6B", dim=300)

    # Preprocess the Visual Genome dataset.
    samples = load_samples(os.path.join(root, "vg_objects.json"))
    samples_preprocessed, categories = preprocess_vg(samples, glove)
    save_samples_preprocessed(
        os.path.join(root, "vg_objects_preprocessed.json"),
        samples_preprocessed
    )

    # Compute mappings from category tokens to GloVe indices.
    cat_mappings = compute_cat_mappings(categories, glove)
    save_cat_mappings(os.path.join(root, "cat_mappings.pkl"), cat_mappings)


if __name__ == "__main__":
    main()
