import argparse
import os

import torch
from image_datasets import CocoImages, VisualGenomeImages, data_transforms
from infer_helpers import calculate_recall
from torch.utils.data import ConcatDataset, DataLoader
from torchtext.vocab import GloVe
from tqdm import tqdm


def baseline(args: argparse.Namespace,
             dataloader: DataLoader,
             glove: GloVe,
             top_20_words: list[str]):
    """
    Perform the baseline experiment on the given dataset.

    Args:
        args (argparse.Namespace): The command line arguments.
        dataloader (DataLoader): The dataloader to be used for inference.
        glove (GloVe): The GloVe embeddings object.
        top_20_words (list[str]): The top 20 words in the dataset.
    """
    # Initialize the recall.
    recall = 0
    recall_terms = 0

    # Iterate over the dataset. Use bacthes for efficiency.
    for imgs, targets, masks in tqdm(dataloader):
        # Move images to GPU.
        imgs, targets, masks = imgs.cuda(), targets.cuda(), masks.cuda()

        for img_idx in range(len(imgs)):
            # Compute the recall for the current image.
            recall_u_i = calculate_recall(
                imgs[img_idx],
                targets[img_idx],
                masks[img_idx],
                {glove.stoi[word] for word in top_20_words[:args.s]},
                torch.ones_like(imgs[img_idx]).bool(),
                args.threshold_iou
            )
            if recall_u_i is not None:
                recall += recall_u_i
                recall_terms += 1

    # Compute the average recall.
    recall /= recall_terms
    print(f"Recall@{args.s}: {recall:.3f}")


def main(args: argparse.Namespace):
    # Set up GloVe word embeddings.
    glove = GloVe(name="6B", dim=args.word_embedding_dim)

    # Set up dataset.
    if args.refer == "coco":
        # Set up COCO dataset.
        root = os.path.join(args.dir_data, "coco")
        img_width, img_height = 224, 224
        datasets = {}
        datasets["train"] = CocoImages(
            ann_file=os.path.join(root,
                                  "annotations/instances_train2017.json"),
            root=os.path.join(root, "train2017"),
            cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
            transform=data_transforms["train"],
            filter_width=img_width,
            filter_height=img_height
        )
        datasets["val"] = CocoImages(
            ann_file=os.path.join(root,
                                  "annotations/instances_val2017.json"),
            root=os.path.join(root, "val2017"),
            cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
            transform=data_transforms["val"],
            filter_width=img_width,
            filter_height=img_height
        )
        dataset = ConcatDataset([datasets["train"], datasets["val"]])

        # The following list was created by the `statistics_coco.py` file.
        top_20_words = [
            "person", "car", "chair", "book", "bottle",
            "cup", "dining", "table", "traffic", "light",
            "bowl", "handbag", "bird", "boat", "truck",
            "bench", "umbrella", "cow", "banana", "backpack"
        ]
    elif args.refer == "vg":
        # Set up Visual Genome dataset.
        root = os.path.join(args.dir_data, "vg")
        img_width, img_height = 224, 224
        dataset = VisualGenomeImages(
            objs_file=os.path.join(root, "vg_objects_preprocessed.json"),
            root=root,
            cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
            transform=data_transforms["val"],
            filter_width=img_width,
            filter_height=img_height
        )

        # The following list was created by the `statistics_vg.py` file.
        top_20_words = [
            "window", "tree", "man", "shirt", "wall",
            "person", "building", "land", "sign", "light",
            "sky", "leg", "hand", "head", "leaf",
            "pole", "grass", "hair", "car", "woman"
        ]
    else:
        raise NotImplementedError(f"Reference dataset '{args.refer}' is "
                                  "not implemented.")

    # Set up dataloader.
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Print confirmation message.
    print()
    print("[ Dataset was loaded successfully! ]".center(79, "-"))
    print()

    # ------------------------------ BASELINE ------------------------------- #

    with torch.no_grad():
        baseline(args, dataloader, glove, top_20_words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for data loading")
    parser.add_argument("--dir-data", type=str, default="./data",
                        help="Path to the datasets")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of subprocesses to use for data loading")
    parser.add_argument("--refer", type=str, default="coco",
                        choices=("vg", "coco"),
                        help="Reference dataset")
    parser.add_argument("--s", type=int, default=5,
                        help="Number of semantics contributed by each top "
                        "activated image")
    parser.add_argument("--threshold-iou", type=float, default=0.04,
                        help="Threshold for filtering out low IoU scores.")
    parser.add_argument("--word-embedding-dim", type=int, default=300,
                        help="GloVe word embedding dimension to use")
    args = parser.parse_args()
    print(f"{args=}")

    main(args)
