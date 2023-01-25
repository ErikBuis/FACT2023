"""Script to preprocess the Visual Genome dataset as described in the paper."""

import json
import os
import pickle
from collections import defaultdict

from torchtext.vocab import GloVe


def compute_cat_mappings(label_file: str, glove: GloVe) \
        -> dict[str, dict]:
    """
    Compute mappings from COCO categories to GloVe indices.

    Args:
        ann_file: Path to the annotation file.
        glove: GloVe object.

    Returns:
        dict: A dictionary containing the mappings. The keys are:
            - "stoi" (dict): COCO category to GloVe index.
            - "itos" (dict): GloVe index to COCO category.
    """
    with open(label_file, "rb") as f:
        vg_labels = pickle.load(f)
    cat_mappings = {"stoi": {}, "itos": {}}
    for token in vg_labels:
        cat_mappings["stoi"][token] = glove.stoi[token]
        cat_mappings["itos"][glove.stoi[token]] = token
    return cat_mappings


def _load_samples(root: str) -> list:
    """Load the object data from the json file."""
    print("Loading object data...", end=" ")
    path = os.path.join(root, "vg_objects.json")
    with open(path) as f:
        samples = json.load(f)
    print("Done.")
    return samples


def _save_samples(root: str, samples: list[dict[str, list[dict[str, int]]]]):
    """Save the preprocessed object data to the json file."""
    print("Saving object data...", end=" ")
    path = os.path.join(root, "vg_objects_preprocessed.json")
    with open(path, "w") as f:
        json.dump(samples, f)
    print("Done.")


def preprocess_vg(samples: list, min_count: int = 100) \
        -> list[dict[str, list[dict[str, int]]]]:
    """
    Preprocess the data as described in the paper.

    Excerpt from the paper:
    "We only use images that have box-able annotations for our experiments.
    During pre-processing, we combined object categories based on their synset
    names, combined instances of the same category in the same image, and
    deleted categories that appear in less than 100 training images. In the
    end, there remains 106,215 out of 107,228 images, 1,208 out of 80,138
    object categories, and at most 47 object categories per image."

    Args:
        samples (list): The samples to preprocess.
        min_count (int, optional): The minimum number of instances of a
            category to keep. Defaults to 100.

    Returns:
        list[dict[str, list[dict[str, int]]]]: The preprocessed samples.
    """
    # Go through the dataset to perform the main preprocessing steps.
    print("Preprocessing data...", end=" ")
    catagory_counts = defaultdict(int)
    preprocessed_samples = []
    for image in samples:
        # Remove images that have no box-able annotations.
        if len(image["objects"]) == 0:
            continue

        result = defaultdict(list)
        for obj in image["objects"]:
            # Combine object categories based on their synset names.
            if len(obj["synsets"]) == 0:
                continue

            # Count the number of instances of each category.
            name = obj["synsets"][0].split(".")[0]
            catagory_counts[name] += 1

            # Combine instances of the same category in the same image.
            result[name].append({"x": obj["x"], "y": obj["y"],
                                 "h": obj["h"], "w": obj["w"]})

        preprocessed_samples.append({"image_id": image["image_id"],
                                     "objects": result})
    print("Done.")

    # Search for categories that appear in less than 100 training images.
    print(f"Searching for categories that appear in less than {min_count} "
          "training images...", end=" ")
    objects_to_remove = [category
                         for category, count in catagory_counts.items()
                         if count < min_count]
    print("Done.")

    # Delete categories that appear in less than 100 training images.
    print(f"Removing {len(objects_to_remove)} categories...")
    for image in preprocessed_samples:
        for category in objects_to_remove:
            image["objects"].pop(category, None)
    preprocessed_samples = [image for image in preprocessed_samples
                            if len(image["objects"]) > 0]
    print("Done.")

    # Print some statistics.
    print()
    print("Preprocessing complete.")
    print("Amount of categories remaining: "
          f"{len(catagory_counts) - len(objects_to_remove)}")
    print("Amount of images remaining: "
          f"{len(preprocessed_samples)}")

    return preprocessed_samples


def main():
    root = "./data/vg"

    # Compute the category mappings.
    label_file = os.path.join(root, "vg_labels.pkl")
    glove = GloVe(name="6B", dim=300)
    print("Computing category mappings...", end=" ")
    cat_mappings = compute_cat_mappings(label_file, glove)
    print("Done.")

    # Save the category mappings into a pickle file.
    with open(os.path.join(root, "cat_mappings.pkl"), "wb") as f:
        pickle.dump(cat_mappings, f)

    # Preprocess the Visual Genome dataset.
    # samples = _load_samples(root)
    # preprocessed_samples = preprocess_vg(samples)
    # _save_samples(root, preprocessed_samples)


if __name__ == "__main__":
    main()
