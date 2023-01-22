"""Script to preprocess the Visual Genome dataset as described in the paper."""

import json
import os
from collections import defaultdict

from tqdm import tqdm


def _load_samples(root: str) -> list:
    """Load the object data from the json file."""
    print("Loading object data...")
    path = os.path.join(root, "vg_objects.json")
    with open(path) as f:
        samples = json.load(f)
    print("Loading complete.")
    
    return samples


def _save_samples(root: str, samples: list[dict[str, list[dict[str, int]]]]):
    """Save the preprocessed object data to the json file."""
    print("Saving object data...")
    path = os.path.join(root, "vg_objects_preprocessed.json")
    with open(path, "w") as f:
        json.dump(samples, f)
    print("Saving complete.")


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
    catagory_counts = defaultdict(int)
    preprocessed_samples = []
    print("Preprocessing data...")
    for image in tqdm(samples):
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

    # Delete categories that appear in less than 100 training images.
    print(f"Removing categories that appear in less than {min_count} "
                 "training images...")
    objects_to_remove = [category
                         for category, count in catagory_counts.items()
                         if count < min_count]
    print(f"Removing {len(objects_to_remove)} categories. "
                 f"{len(catagory_counts) - len(objects_to_remove)} remain.")
    for image in tqdm(preprocessed_samples):
        for category in objects_to_remove:
            image["objects"].pop(category, None)
    preprocessed_samples = [image for image in preprocessed_samples
                            if len(image["objects"]) > 0]
    print("Preprocessing complete. "
                 f"{len(preprocessed_samples)} images remain.")

    return preprocessed_samples


def main():
    root = "./data/vg"
    samples = _load_samples(root)
    preprocessed_samples = preprocess_vg(samples)
    _save_samples(root, preprocessed_samples)


if __name__ == "__main__":
    main()
