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


def find_max_activations(args: argparse.Namespace, model: nn.Module,
                         dataloader: DataLoader, dir_max_acts: str) \
        -> dict[int, torch.Tensor]:
    """
    For each target filter, find the top p images that cause it to output the
    highest activation.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        dataloader (Dataloader): The dataloader for the reference dataset.
        dir_max_acts (str): The directory to save the max activations to.

    Returns:
        dict[int, torch.Tensor]: For each filter u, a sorted list of indices of
            the top p images that let the filter output the highest activation.
    """
    # Check whether the max activations have already been computed.
    max_imgs_sorted = {u: None for u in args.u}
    for u in args.u:
        path_max_acts = os.path.join(dir_max_acts, f"p-{args.p}_u-{u}.pt")
        if os.path.exists(path_max_acts):
            print(f"Found precomputed max activations in '{path_max_acts}'. "
                  "Loading...", end=" ")
            max_imgs_sorted[u] = torch.load(path_max_acts)
            print("Done.")
    if all(max_imgs_sorted[u] is not None for u in args.u):
        return max_imgs_sorted

    # For each target filter, save a sorted list of (-max_act, img_idx) tuples,
    # i.e. `max_acts_sorted[u] = [(-max_act, img_idx), ...]`.
    print("Computing max activations...")
    u_not_computed = [u for u in args.u if max_imgs_sorted[u] is None]
    max_acts_sorted = {u: [] for u in u_not_computed}
    for batch_idx, (imgs, _, _) in enumerate(tqdm(dataloader)):
        # Move batch data to GPU.
        imgs = imgs.cuda().detach()

        # Forward pass through feature extractor Feat().
        acts = forward_Feat(args, model, imgs)

        # Only select the filters we are interested in.
        acts = acts[:, u_not_computed, :, :]

        # Get the max activation of each image on each target filter.
        max_acts_batch = acts.amax(dim=(-1, -2))

        # Efficiently insert the max activations into the sorted lists. Use
        # binary search to iteratively insert images that cause each filter to
        # output the highest max activation into the top p.
        for img_idx, max_acts_img in \
                enumerate(max_acts_batch, start=batch_idx * args.batch_size):
            for u, max_act in zip(u_not_computed, max_acts_img):
                t = (-max_act.item(), img_idx)
                insertion_idx = bisect_right(max_acts_sorted[u], t)
                if insertion_idx < args.p:
                    max_acts_sorted[u].insert(insertion_idx, t)
                    if len(max_acts_sorted[u]) > args.p:
                        max_acts_sorted[u].pop()
    print("Done.")

    # Extract the indices of the top p images.
    for u in u_not_computed:
        max_imgs_sorted[u] = torch.tensor([t[1] for t in max_acts_sorted[u]])
        path_max_acts = os.path.join(dir_max_acts, f"p-{args.p}_u-{u}.pt")
        print(f"Saving max activations to '{path_max_acts}'...", end=" ")
        torch.save(max_imgs_sorted[u], path_max_acts)
        print("Done.")

    return max_imgs_sorted


def find_thresholds_act_masking(args: argparse.Namespace, model: nn.Module,
                                dataloader: DataLoader,
                                dir_thresholds_act_masking: str) \
        -> dict[int, torch.Tensor]:
    """
    Find the threshold for each target filter such that the probability of an
    activation being greather than the threshold is equal to 0.005.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        dataloader (Dataloader): The dataloader for the reference dataset.
        dir_thresholds_act_masking (str): The directory to save the thresholds
            to.

    Returns:
        dict[int, torch.Tensor]: For each filter u, the threshold such that the
            probability of an activation being greather than the threshold is
            equal to 0.005.
    """
    # Check whether the thresholds have already been computed.
    thresholds_act_masking = {u: None for u in args.u}
    for u in args.u:
        path_thresholds_act_masking = os.path.join(dir_thresholds_act_masking,
                                                   f"u-{u}.pt")
        if os.path.exists(path_thresholds_act_masking):
            print(f"Found precomputed thresholds in "
                  f"'{path_thresholds_act_masking}'. Loading...", end=" ")
            thresholds_act_masking[u] = torch.load(path_thresholds_act_masking)
            print("Done.")
    if all(thresholds_act_masking[u] is not None for u in args.u):
        return thresholds_act_masking

    # For each target filter, save all activations in a list.
    print("Computing thresholds...")
    u_not_computed = [u for u in args.u if thresholds_act_masking[u] is None]
    acts_all = {u: torch.tensor([], device="cuda") for u in u_not_computed}
    for imgs, _, _ in tqdm(dataloader):
        # Move batch data to GPU.
        imgs = imgs.cuda().detach()

        # Forward pass through feature extractor Feat().
        acts = forward_Feat(args, model, imgs)

        # Save the activations.
        for u in u_not_computed:
            acts_all[u] = torch.cat((acts_all[u],
                                     torch.flatten(acts[:, u, :, :])))
    print("Done.")

    # Compute the thresholds.
    for u in u_not_computed:
        thresholds_act_masking[u] = torch.quantile(torch.sort(acts_all[u])[0],
                                                   0.005)
        path_thresholds_act_masking = os.path.join(dir_thresholds_act_masking,
                                                   f"u-{u}.pt")
        print(f"Saving thresholds to '{path_thresholds_act_masking}'...",
              end=" ")
        torch.save(thresholds_act_masking[u], path_thresholds_act_masking)
        print("Done.")

    return thresholds_act_masking


def explain(method: str, model: nn.Module, imgs: torch.Tensor,
            acts: torch.Tensor, acts_u: torch.Tensor,
            acts_u_resized: torch.Tensor, threshold_act_masking: float) \
        -> torch.Tensor:
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
        threshold_act_masking (float): The threshold for masking activations.

    Returns:
        torch.Tensor: The prediction of the model for each image.
            Shape: [batch_size, word_embedding_dim]
    """
    if method == "original":
        # Original image.
        pass
    if method == "image":
        # Image masking.
        imgs *= acts_u_resized > threshold_act_masking
        acts = forward_Feat(args, model, imgs)
    elif method == "activation":
        # Activation masking.
        acts *= acts_u > threshold_act_masking
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
              embeddings: torch.Tensor,
              dir_heatmaps: str,
              dir_max_acts: str,
              dir_thresholds_act_masking: str) -> Optional[torch.Tensor]:
    """
    Perform inference on the given trained model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        dataloader (DataLoader): The dataloader to be used for inference.
        glove (GloVe): The GloVe embeddings object.
        embeddings (torch.Tensor): The ground-truth category embeddings.
            Shape: [word_embedding_dim, num_categories].
        dir_heatmaps (str): The directory to save the heatmaps to.
        dir_max_acts (str): The directory to save the max activations to.
        dir_thresholds_act_masking (str): The directory to save the thresholds
            for activation masking to.
    """
    # Set model to evaluation mode.
    model.eval()

    # For each filter u, look for the p images that activate it the most.
    max_imgs_sorted = find_max_activations(args, model, dataloader,
                                           dir_max_acts)

    # For each filter u, find the threshold for activation masking.
    thresholds_act_masking = find_thresholds_act_masking(
        args, model, dataloader, dir_thresholds_act_masking
    )

    # Create a table where each row contains the top tokens that were
    # accociated with a specific target filter.
    headers = ["Filter"] + [str(i) for i in range(1, args.num_tokens+1)]
    table = []

    # Initialize the recall.
    recall = 0
    recall_terms = 0

    # Loop over all target filters to explain each one.
    for u, max_imgs_sorted_u in max_imgs_sorted.items():
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
        for batch_idx, (imgs, targets, masks) in enumerate(dataloader_u):
            # Move batch data to GPU.
            imgs = imgs.cuda().detach()
            targets, masks = targets.cuda().detach(), masks.cuda().detach()

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
            preds = explain(args.method, model, imgs, acts, acts_u,
                            acts_u_resized, thresholds_act_masking[u])

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
            for img_idx in range(len(imgs)):
                if batch_idx * args.batch_size + img_idx \
                        >= args.num_heatmaps:
                    break

                heatmaps.append(create_heatmap(unnormalize(imgs[img_idx]),
                                               acts_u_resized[img_idx]))

                # Compute IoU between the heatmap and the mask.
                W_u_i = [glove_idx.item()
                         for glove_idx in word_preds_per_img[img_idx]]
                R_x = acts_u_resized[img_idx] > thresholds_act_masking[u]

                masks_img_resized = F.interpolate(masks[img_idx],
                                                  size=imgs.shape[-2:],
                                                  mode="nearest").bool()
                targets_img = targets[img_idx]

                print("----------------------------------------")
                print(f"Predicted words: {[glove.itos[idx] for idx in W_u_i]}")

                G_u_i = []
                for M_j, t_j in zip(masks_img_resized, targets_img):
                    IoU = (R_x & M_j).sum() / (R_x | M_j).sum()
                    if IoU > args.threshold_iou:
                        G_u_i.append(t_j.item())
                        print(f"Ground-truth word: {glove.itos[t_j.item()]}")
                if len(G_u_i) == 0:
                    continue

                print(f"{set(G_u_i)=}")
                print(f"{set(W_u_i)=}")
                result_g = set(G_u_i) & set(W_u_i)
                print(f"Set: {result_g}")
                print(f"Length of intersection: {len(result_g)}")
                recall_u_i = len(set(G_u_i) & set(W_u_i)) / len(G_u_i)
                recall += recall_u_i
                recall_terms += 1

        # Sort the predicteded words by their frequencies.
        words, counts = torch.unique(word_preds, return_counts=True)
        words = words[torch.argsort(counts, descending=True)[:args.num_tokens]]

        # Convert the word indices to word tokens.
        tokens = [glove.itos[word] for word in words]
        table.append([u] + tokens + ["-"] * (len(headers) - len(tokens) - 1))

        # Visualize the activation heatmaps and save them to disk.
        heatmaps_grid = torchvision.utils.make_grid(heatmaps)
        path_heatmaps = os.path.join(
            dir_heatmaps,
            f"method-{args.method}_p-{args.p}_u-{u}.png"
        )
        torchvision.utils.save_image(heatmaps_grid, path_heatmaps)

        # Log the heatmaps to wandb.
        if args.wandb:
            caption = " | ".join((f"Method: {args.method}",
                                  f"Concept: {tokens[0]}",
                                  f"Filter: {u}"))
            images = wandb.Image(heatmaps_grid, caption=caption)
            wandb.log({"Activation heatmaps with associated concept": images})

    # Visualize the table of filter explanations.
    print(tabulate(table, headers=headers, tablefmt="github"))

    # Log the table of filter explanations to wandb.
    if args.wandb:
        table = wandb.Table(data=table, columns=headers)
        wandb.log({f"Top-{args.num_tokens} filter explanations": table})

    # Compute the average recall.
    recall /= recall_terms
    print(f"Average recall: {recall}")


def main(args: argparse.Namespace):
    """
    Perform inference on a trained model.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Set up output path.
    args.dir_save = os.path.join(args.dir_save, args.name)
    if not os.path.exists(args.dir_save):
        raise FileNotFoundError(f"Could not find '{args.dir_save}'. "
                                "Please train a model first.")

    # Create a directory to save the heatmaps.
    dir_heatmap = os.path.join(args.dir_save, "heatmaps")
    if not os.path.exists(dir_heatmap):
        os.makedirs(dir_heatmap)

    # Create a directory to save the maximum activations.
    dir_max_acts = os.path.join(args.dir_save, "max_acts")
    if not os.path.exists(dir_max_acts):
        os.makedirs(dir_max_acts)

    # Create a directory to save the thresholds for masking out activations.
    dir_thresholds_act_masking = os.path.join(args.dir_save,
                                              "thresholds_act_masking")
    if not os.path.exists(dir_thresholds_act_masking):
        os.makedirs(dir_thresholds_act_masking)

    # Set up wandb logging.
    if args.wandb:
        path_wandb_id_file = pathlib.Path(os.path.join(args.dir_save,
                                                       "runid.txt"))
        if path_wandb_id_file.exists():
            resume_id = path_wandb_id_file.read_text()
            wandb.init(project="temporal_scale", resume=resume_id,
                       name=args.name, config=args)
        else:
            print("Creating new wandb instance...", path_wandb_id_file)
            run = wandb.init(project="temporal_scale",
                             name=args.name, config=args)
            path_wandb_id_file.write_text(str(run.id))
        wandb.config.update(args)

    # Set up GloVe word embeddings.
    glove = GloVe(name="6B", dim=args.word_embedding_dim)
    embeddings = glove.vectors.T.cuda()

    # Set up dataset.
    if args.refer == "vg":
        # Set up Visual Genome dataset.
        root = os.path.join(args.dir_data, "vg")
        dataset = VisualGenomeImages(
            root=root,
            objs_file=os.path.join(root, "vg_objects_preprocessed.json"),
            cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
            transform=data_transforms["val"]
        )
    elif args.refer == "coco":
        # Set up COCO dataset.
        root = os.path.join(args.dir_data, "coco")
        dataset = CocoImages(
            root=os.path.join(root, "val2017"),
            ann_file=os.path.join(root, "annotations/instances_val2017.json"),
            cat_mappings_file=os.path.join(root, "cat_mappings.pkl"),
            transform=data_transforms["val"],
        )
    else:
        raise NotImplementedError(f"Reference dataset '{args.refer}' "
                                  "not implemented.")

    # Set up dataloader.
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Set up model.
    model = setup_explainer(args, args.random)
    if args.path_model is None:
        args.path_model = os.path.join(args.dir_save, "ckpt_best.pth.tar")
    if not os.path.exists(args.path_model):
        raise FileNotFoundError(f"No model found at '{args.path_model}'.")

    print(f"Loading model from '{args.path_model}'...", end=" ")
    model.load_state_dict(torch.load(args.path_model)["state_dict"])
    model = model.cuda()
    print("Done.")

    # Print confirmation message.
    print()
    print("[ Model was loaded successfully! ]".center(79, "-"))
    print()

    # ------------------------------ INFERENCE ------------------------------ #

    with torch.no_grad():
        inference(args, model, dataloader, glove, embeddings,
                  dir_heatmap, dir_max_acts, dir_thresholds_act_masking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size data loading")
    parser.add_argument("--dir-data", type=str, default="./data",
                        help="Path to the datasets")
    parser.add_argument("--dir-save", type=str, default="./outputs",
                        help="Path to model checkpoints")
    parser.add_argument("--layer-target", type=str, default="layer4",
                        help="Target layer to explain")
    parser.add_argument("--layer-classifier", type=str, default="fc",
                        help="Name of classifier layer")
    parser.add_argument("--method", type=str, default="projection",
                        choices=("original", "image",
                                 "activation", "projection"),
                        help="Method used to explain the target filter")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="Target network")
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
                        "each filter")
    parser.add_argument("--path-model", type=str, default=None,
                        help="Path to trained explainer model")
    parser.add_argument("--random", action="store_true",
                        help="Use a randomly initialized target model instead "
                        "of torchvision pretrained weights")
    parser.add_argument("--refer", type=str, default="coco",
                        choices=("vg", "coco"),
                        help="Reference dataset")
    parser.add_argument("--s", type=int, default=5,
                        help="Number of semantics contributed by each top "
                        "activated image")
    parser.add_argument("--threshold-iou", type=float, default=0.04,
                        help="Threshold for filtering out low IoU scores.")
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
    print(args, "\n")

    main(args)
