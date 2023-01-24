"""Script to preprocess the COCO dataset as described in the paper."""

import os
import pickle
import torch

from pycocotools.coco import COCO
from torchtext.vocab import GloVe


def compute_cat_mappings(ann_file: str, glove: GloVe) \
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
    coco = COCO(ann_file)
    cat_mappings = {"stoi": {}, "itos": {}}
    for cat in coco.cats.values():
        for token in cat["name"].split(" "):
            cat_mappings["stoi"][token] = glove.stoi[token]
            cat_mappings["itos"][glove.stoi[token]] = token
    return cat_mappings


def main():
    root = "./data/coco"

    # Compute the category mappings.
    ann_file = os.path.join(root, "annotations/instances_train2017.json")
    glove = GloVe(name="6B", dim=300)
    print("Computing category mappings...", end=" ")
    cat_mappings = compute_cat_mappings(ann_file, glove)
    print("Done.")

    # Save the category mappings into a pickle file.
    with open(os.path.join(root, "cat_mappings.pkl"), "wb") as f:
        pickle.dump(cat_mappings, f)


if __name__ == "__main__":
    main()
