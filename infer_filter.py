import argparse
import os
import pathlib
from bisect import bisect_right
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from image_datasets import (CocoImages, VisualGenomeImages, data_transforms,
                            unnormalize)
from model_loader import forward_Exp, forward_Feat, setup_explainer
from tabulate import tabulate
from torch.linalg import vector_norm
from torch.utils.data import DataLoader, Subset
from torchtext.vocab import GloVe
from tqdm import tqdm


def find_images_max_activations(args: argparse.Namespace, model: nn.Module,
                                dataloader: DataLoader) -> torch.Tensor:
    """
    For each target filter, find the top p images that cause it to output the
    highest activation.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        dataloader (Dataloader): The dataloader for the reference dataset.

    Returns:
        torch.Tensor: For each filter u, a sorted list of indices of the top p
            images that cause the filter to output the highest activation.
            Shape: [amount_target_filters, p].
    """
    # Check if the max activations have already been computed.
    path = os.path.join(args.save_dir, f"max_activations/max_activations_{len(args.u)}_{args.p}.pt")
    if os.path.exists(path):
        print(f"Found precomputed max activations at {path}, loading..")
        return torch.load(args.max_activations_path)

    # For each target filter, save a sorted list of (-max_act, img_idx) tuples.
    # max_acts_sorted[u] = [(-max_act, img_idx), ...]
    max_acts_sorted = [[] for _ in range(len(args.u))]

    for batch_idx, (imgs, _, _) in enumerate(tqdm(dataloader)):
        # Move batch data to GPU.
        imgs = imgs.cuda().detach()

        # Forward pass through feature extractor Feat().
        acts = forward_Feat(args, model, imgs)

        # Only select the filters we are interested in.
        acts = acts[:, args.u, :, :]

        # Get the max activation of each image on each target filter.
        max_acts_batch = acts.amax(dim=(-1, -2))

        # Efficiently insert the max activations into the sorted lists.
        for img_idx, max_acts_img in enumerate(max_acts_batch,
                                               batch_idx * args.batch_size):
            for max_act, max_acts_filter in zip(max_acts_img, max_acts_sorted):
                tup = (-max_act.item(), img_idx)
                insertion_idx = bisect_right(max_acts_filter, tup)
                if insertion_idx < args.p:
                    max_acts_filter.insert(insertion_idx, tup)
                    if len(max_acts_filter) > args.p:
                        max_acts_filter.pop()

    # We only need the image indices, so we will extract those here.
    max_activations = torch.tensor([[tup[1] for tup in max_act_imgs] 
                                    for max_act_imgs in max_acts_sorted], dtype=torch.long)
                        
    # Create the directory if it does not exist.
    pathlib.Path(os.path.join(args.save_dir, "max_activations")).mkdir(parents=True, exist_ok=True)
    # Save the max activations.
    print(f"Saving max activations to {path}..")
    torch.save(max_activations, os.path.join(
        args.save_dir, f"max_activations/max_activations_{len(args.u)}_{args.p}.pt"))
    
    return max_activations


def explain(method: str, model: nn.Module, imgs: torch.Tensor,
            acts: torch.Tensor, acts_u: torch.Tensor,
            acts_u_resized: torch.Tensor) -> torch.Tensor:
    """
    Explain the given image using the given method.

    Args:
        method (str): The method to use for the explanation.
        model (nn.Module): Feat() + Exp() model.
        imgs (torch.Tensor): The batch of images to use in the explanation.
            Shape: [batch_size, 3, 224, 224].
        acts (torch.Tensor): The activations of all filters.
            Shape: [batch_size, amount_filters, 7, 7].
        acts_u (torch.Tensor): The activations of the target filter.
            Shape: [batch_size, 1, 7, 7].
        acts_u_resized (torch.Tensor): The activations of the target filter,
            resized to the size of the image.
            Shape: [batch_size, 1, 224, 224].

    Returns:
        torch.Tensor: The prediction of the model for each image.
            Shape: [batch_size, word_embedding_dim]
    """
    if method == "original":
        # Original image.
        pass
    if method == "image":
        # Image masking.
        imgs *= acts_u_resized > args.mask_threshold
        acts = forward_Feat(args, model, imgs)
    elif method == "activation":
        # Activation masking.
        acts *= acts_u > args.mask_threshold
    elif method == "projection":
        # Filter attention projection.
        # TODO Check whether this is correct. The paper says:
        # TODO "The input to the explainer, Exp(), with respect to a target
        # TODO filter u, is computed as follows: F_k^att = a(F_u, F_k) * Fk
        # TODO for all k, where a() computes spatial correlations between
        # TODO filters by cosine similarity."
        # TODO But the code seems to be different. Here, we seem to simply be
        # TODO computing an element-wise product.
        acts *= acts_u / vector_norm(acts_u, dim=(-2, -1), keepdim=True)
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

    # Forward pass through feature explainer Exp().
    return forward_Exp(args, model, acts)


def create_heatmap(img: torch.Tensor, acts_u_resized: torch.Tensor,
                   opacity: float = 0.3) -> torch.Tensor:
    """
    Combine an image and an activation map to create a heatmap.

    Args:
        img (torch.Tensor): Image to visualize.
            Shape: [3, 224, 224].
        acts_u_resized_img (torch.Tensor): Activation map of the target filter
            for the given image.
            Shape: [1, 224, 224].
        opacity (float, optional): Opacity of the heatmap.
            Defaults to 0.3.

    Returns:
        torch.Tensor: Activation heatmap of the image.
            Shape: [3, 224, 224].
    """
    return img * (acts_u_resized / acts_u_resized.max()).clamp(min=opacity)


def inference(args: argparse.Namespace,
              model: nn.Module,
              dataloader: DataLoader,
              glove: GloVe,
              embeddings: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Perform inference on the given trained model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        dataloader (DataLoader): The dataloader to be used for inference.
        glove (GloVe): The GloVe embeddings object.
        embeddings (torch.Tensor): The ground-truth category embeddings.
            Shape: [word_embedding_dim, num_categories].
    """
    # Set model to evaluation mode.
    model.eval()

    # For each filter u, search for the p images that cause the maximum value
    # of u's activation map to be the highest.
    print("Extracting max activations...")
    max_imgs_sorted = find_images_max_activations(args, model, dataloader)
    print("Done extracting max activations.")

    # Create a table where each row contains the top tokens that were
    # accociated with a specific target filter.
    headers = ["Filter"] + [str(i) for i in range(1, args.num_tokens+1)]
    table = []

    # Create a directory to save the heatmaps.
    heatmap_dir = os.path.join(args.save_dir, "heatmaps")
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)

    # Loop over all target filters to explain each one.
    for u, max_imgs_sorted_u in zip(args.u, max_imgs_sorted):
        print(f"Interpreting filter {u} with top {args.p} activated images...")

        # Create a new dataloader that only contains the p images that
        # activated the target filter the most.
        dataset_u = Subset(dataloader.dataset, max_imgs_sorted_u)
        dataloader_u = DataLoader(dataset_u, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

        # Initialize the list of heatmaps and the list of word predictions.
        heatmaps = []
        word_preds = None

        # Iterate over the p images that activated the target filter the
        # most. Use batches for efficiency.
        for batch_idx, (imgs, _, _) in tqdm(enumerate(dataloader_u)):
            # Move batch data to GPU.
            imgs = imgs.cuda().detach()

            # Forward pass through feature extractor Feat().
            acts = forward_Feat(args, model, imgs)

            # Get the activation maps of the target filter.
            acts_u = acts[:, [u], :, :]

            # Resize the activation map to the size of the image.
            acts_u_resized = F.interpolate(acts_u, size=imgs.shape[-2:],
                                           mode="bilinear")

            # Skip images that do not activate the filter.
            max_acts_u = torch.amax(acts_u, dim=(-3, -2, -1))
            imgs_no_activation = max_acts_u <= 0
            if imgs_no_activation.any():
                imgs = imgs[~imgs_no_activation]
                acts = acts[~imgs_no_activation]
                acts_u = acts_u[~imgs_no_activation]
                acts_u_resized = acts_u_resized[~imgs_no_activation]
                max_acts_u = max_acts_u[~imgs_no_activation]

            # Explain the filters with the batch of images.
            preds = explain(args.method, model, imgs, acts, acts_u, acts_u_resized)

            # Compute the cosine similarity between each prediction and
            # each ground-truth word embedding. Then sort the results
            # in descending order, so that the top `s` indices represent
            # the words that are most similar to the model's prediction.
            word_preds_per_img = torch.argsort(
                (preds @ embeddings)
                / (vector_norm(preds, dim=1, keepdim=True) @
                   vector_norm(embeddings, dim=0, keepdim=True)),
                dim=1,
                descending=True
            )[:, :args.s]

            # Repeat the `s` word predictions if the image that created
            # them caused high activations in our target filter to occur.
            # The assumption is that these images are more relevant to
            # how the filter should be interpreted.
            if args.weigh_s_by_relevance:
                word_preds_per_img = torch.repeat_interleave(
                    word_preds_per_img,
                    max_acts_u.int(),
                    dim=0
                )

            # Concatenate the `batch_size * s` word predictions we just
            # made with all the other word predictions we have made so far.
            word_preds = word_preds_per_img if word_preds is None \
                else torch.cat((word_preds, word_preds_per_img))

            # Visualize activation heatmaps for the top `num_heatmaps` images.
            if args.wandb:
                for img_idx in range(len(imgs)):
                    if batch_idx * args.batch_size + img_idx \
                            >= args.num_heatmaps:
                        break
                    heatmaps.append(create_heatmap(unnormalize(imgs[img_idx]),
                                                   acts_u_resized[img_idx]))

        # Sort the predicteded words by their frequencies.
        words, counts = torch.unique(word_preds, return_counts=True)
        words = words[torch.argsort(counts, descending=True)[:args.num_tokens]]

        # Convert the word indices to word tokens.
        with open(os.path.join(args.data_dir, "entities.txt"), "r") as f:
            entities = [line.strip() for line in f if line != "\n"]
        tokens = [glove.itos[word] for word in words
                  if glove.itos[word] in entities]
        table.append([u] + tokens + ["-"] * (len(headers) - len(tokens) - 1))

        if args.wandb:
            # Log the heatmaps to wandb.
            heatmap_grid = torchvision.utils.make_grid(heatmaps)
            caption = " | ".join((f"Method: {args.method}",
                                  f"Filter: {u}",
                                  f"Concept: {tokens[0]}"))
            images = wandb.Image(heatmap_grid, caption=caption)
            wandb.log({"Activation heatmaps with associated concept": images})
        else:
            # Save the heatmaps to disk.
            heatmap_path = os.path.join(heatmap_dir, f"filter_{u}.png")
            torchvision.utils.save_image(heatmap_grid, heatmap_path)

    if args.wandb:
        # Log the table of filter explanations to wandb.
        table = wandb.Table(data=table, columns=headers)
        wandb.log({f"Top-{args.num_tokens} filter explanations": table})
    else:
        # Pretty print the table of filter explanations.
        print(tabulate(table, headers=headers, tablefmt="github"))


def main(args: argparse.Namespace):
    """
    Perform inference on a trained model.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Set up output path.
    args.save_dir = os.path.join(args.save_dir, args.name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Store the command line arguments in a text file.
    with open(os.path.join(args.save_dir, "infer_filter_args.txt"), "w") as f:
        f.write(str(args))

    # Set up wandb logging.
    if args.wandb:
        wandb_id_file_path = pathlib.Path(os.path.join(args.save_dir,
                                                       "runid.txt"))
        if wandb_id_file_path.exists():
            resume_id = wandb_id_file_path.read_text()
            wandb.init(project="temporal_scale", resume=resume_id,
                       name=args.name, config=args)
        else:
            print("Creating new wandb instance...", wandb_id_file_path)
            run = wandb.init(project="temporal_scale",
                             name=args.name, config=args)
            wandb_id_file_path.write_text(str(run.id))
        wandb.config.update(args)

    # Set up GloVe word embeddings.
    glove = GloVe(name="6B", dim=args.word_embedding_dim)
    embeddings = glove.vectors.T.cuda()

    # Set up dataset.
    if args.refer == "vg":
        # Set up Visual Genome dataset.
        root = os.path.join(args.data_dir, "vg")
        dataset = VisualGenomeImages(
            root=root,
            transform=data_transforms["val"]
        )
    elif args.refer == "coco":
        # Set up COCO dataset.
        root = os.path.join(args.data_dir, "coco")
        dataset = CocoImages(
            root=os.path.join(root, "val2017"),
            annFile=os.path.join(root, "annotations/instances_val2017.json"),
            cat_mappings_file=os.path.join(root, "coco_label_embedding.pth"),
            transform=data_transforms["val"]
        )
    else:
        raise NotImplementedError(f"Reference dataset '{args.refer}' "
                                  "not implemented.")

    # Set up dataloader.
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Set up model.
    model = setup_explainer(args, random_feature=args.random)
    if args.model_path is None:
        args.model_path = os.path.join(args.save_dir, "ckpt_best.pth.tar")
    if os.path.exists(args.model_path):
        print(f"Loading model from '{args.model_path}'...")
        model.load_state_dict(torch.load(args.model_path)["state_dict"])
    else:
        raise FileNotFoundError(f"No model found at '{args.model_path}'.")
    model = model.cuda()

    # Print confirmation message.
    print()
    print("[ Model was loaded successfully! ]".center(79, "-"))
    print()

    # ------------------------------ INFERENCE ------------------------------ #

    with torch.no_grad():
        inference(args, model, dataloader, glove, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size data loading")
    parser.add_argument("--classifier-name", type=str, default="fc",
                        help="Name of classifier layer")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to the dataset directory")
    parser.add_argument("--disable-save-heatmaps", action="store_false",
                        help="Whether to save heatmaps to wandb")
    parser.add_argument("--layer", type=str, default="layer4",
                        help="Target layer to explain")
    parser.add_argument("--mask-threshold", type=float, default=0.04,
                        help="Threshold for masking out low activations. "
                        "The default value ensures that the probability of "
                        "an activation being above the threshold is 0.005.")
    parser.add_argument("--method", type=str, default="projection",
                        choices=("original", "image",
                                 "activation", "projection"),
                        help="Method used to explain the target filter")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="Target network")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained explainer model")
    parser.add_argument("--name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--num-heatmaps", type=int, default=5,
                        help="Number of activation heatmaps to visualize")
    parser.add_argument("--num-tokens", type=int, default=10,
                        help="Number of tokens to output to explain each "
                        "target filter")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of subprocesses to use for data loading")
    parser.add_argument("--p", type=int, default=25,
                        help="Number of top activated images used to explain "
                        "each filter")  # If filter projection is used.
    parser.add_argument("--random", action="store_true",
                        help="Use a randomly initialized target model instead "
                        "of torchvision pretrained weights")
    parser.add_argument("--refer", type=str, default="coco",
                        choices=("vg", "coco"),
                        help="Reference dataset")
    parser.add_argument("--s", type=int, default=5,
                        help="Number of semantics contributed by each top "
                        "activated image")  # If filter projection is used.
    parser.add_argument("--save-dir", type=str, default="./outputs",
                        help="Path to model checkpoints")
    parser.add_argument("--u", type=list, default=list(range(0, 550, 50)),
                        help="List of indices of the target filters")
    parser.add_argument("--wandb", action="store_true",
                        help="Use wandb for logging")
    # TODO This argument was effectively always true in the original code.
    # TODO Why is this the case? This seems to be an inconsitency between the
    # TODO paper and the code.
    parser.add_argument("--weigh-s-by-relevance", action="store_true",
                        help="For each image, multiply --s by how much the "
                        "image activates the target filter")
    parser.add_argument("--word-embedding-dim", type=int, default=300,
                        help="GloVe word embedding dimension to use")

    args = parser.parse_args()
    print(args)

    main(args)
