import argparse
import os
import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from image_datasets import (CocoInstances, VisualGenomeInstances,
                            data_transforms)
from model_loader import forward_Exp, forward_Feat, setup_explainer
from torch.linalg import vector_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.vocab import GloVe
from train_helpers import CSMRLoss, set_bn_eval


def set_seed(seed: int):
    """
    Set a seed for all random number generators.

    Args:
        seed (int): The seed to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(args: argparse.Namespace,
                    model: nn.Module,
                    loss_fn: nn.Module,
                    train_loader: DataLoader,
                    embeddings: torch.Tensor,
                    epoch: int,
                    optimizer: torch.optim.Optimizer,
                    train_label_indices: Optional[np.ndarray] = None,
                    ks: list[int] = [1, 5, 10, 20]) -> Tuple[float, float]:
    """
    Train one epoch of the model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        loss_fn (nn.Module): The loss function to use.
        train_loader (DataLoader): The training set dataloader.
        embeddings (torch.Tensor): The ground-truth category embeddings.
            Shape: [word_embedding_dim, num_categories].
        epoch (int): The current epoch.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_label_indices (Optional[np.ndarray], optional): The indices of
            the labels to train on. Defaults to None.
            Shape: [num_train_labels].
        ks (list[int], optional): List of k-values. Each k represents the
            number of top category predictions to consider for accuracy
            calculation. Defaults to [1, 5, 10, 20].

    Returns:
        Tuple[float, float]:
            - The average training loss per sample.
            - The average training accuracy@1 per sample.
    """
    # Set model to training mode.
    model.train()
    model.apply(set_bn_eval)

    # Initialize variables.
    amount_batches = len(train_loader)
    ks = sorted(set(ks).union((1,)))
    loss_train = 0
    correct_train = {k: 0 for k in ks}

    # Iterate over the training set.
    for batch_idx, (imgs, targets, masks) in enumerate(train_loader, start=1):
        # Move batch data to GPU.
        imgs = imgs.cuda()
        targets, masks = targets.squeeze(0).cuda(), masks.squeeze(0).cuda()

        # Forward pass.
        acts = forward_Feat(args, model, imgs)
        if torch.sum(masks) > 0:
            acts *= masks
        preds = forward_Exp(args, model, acts)

        # Calculate loss.
        loss = loss_fn(preds, targets, embeddings, train_label_indices)

        # Backward propagation.
        optimizer.zero_grad()
        if loss.requires_grad:
            print("Loss requires grad.")
            loss.backward()
            optimizer.step()

        # Update loss.
        loss_train += loss.data.detach().item()

        # Compute the cosine similarity between each prediction and
        # each ground-truth category embedding. Then sort the results
        # in descending order, so that the top `k` indices represent
        # the categories that are most similar to the model's prediction.
        cat_preds_per_sample = torch.argsort(
            (preds @ embeddings)
            / (vector_norm(preds, dim=1, keepdim=True) @
               vector_norm(embeddings, dim=0, keepdim=True)),
            dim=1,
            descending=True
        )[:, :max(ks)]

        # Calculate accuracy.
        for k in ks:
            for img_idx, cat_preds in enumerate(cat_preds_per_sample):
                correct_train[k] += targets[img_idx, cat_preds[:k]] \
                    .any().detach().item()

        # Print logging data.
        if batch_idx % 10 == 0 or batch_idx == amount_batches:
            epoch_chars = len(str(args.epochs))
            batch_chars = len(str(amount_batches))
            print(f"[Epoch {epoch:{epoch_chars}d}: "
                  f"{batch_idx:{batch_chars}d}/{amount_batches} "
                  f"({int(batch_idx / amount_batches * 100):3d}%)] "
                  f"Loss: {loss.cpu().item():.6f}")
            if args.wandb:
                wandb.log({"Iter_Train_Loss": loss})

        # Save checkpoint.
        if batch_idx % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                },
                os.path.join(args.save_dir, "ckpt_tmp.pth.tar")
            )

        # Free memory.
        torch.cuda.empty_cache()

    # Calculate average loss and average accuracy per sample.
    loss_train /= len(train_loader)
    accs_train = {k: correct_train[k] / len(train_loader.dataset) * 100
                  for k in ks}
    print()
    print(f"Train average loss: {loss_train:.6f}")
    for k in ks:
        print(f"Train top-{k} accuracy: {accs_train[k]:.2f}%")

    return loss_train, accs_train[1]


def validate(args: argparse.Namespace,
             model: nn.Module,
             loss_fn: nn.Module,
             valid_loader: DataLoader,
             embeddings: torch.Tensor,
             train_label_indices: Optional[np.ndarray] = None,
             ks: list[int] = [1, 5, 10, 20]) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model (nn.Module): Feat() + Exp() model.
        loss_fn (nn.Module): The loss function to use.
        valid_loader (DataLoader): The validation set data loader.
        embeddings (torch.Tensor): The ground-truth category embeddings.
            Shape: [word_embedding_dim, num_categories].
        train_label_indices (Optional[np.ndarray], optional): The indices of
            the labels to train on. Defaults to None.
            Shape: [num_train_labels].
        ks (list[int], optional): List of k-values. Each k represents the
            number of top category predictions to consider for accuracy
            calculation. Defaults to [1, 5, 10, 20].

    Returns:
        Tuple[float, float]:
            - The average validation loss per sample.
            - The average validation accuracy@1 per sample.
    """
    # Set model to evaluation mode.
    model.eval()

    # Initialize variables.
    ks = sorted(set(ks).union((1,)))
    loss_valid = 0
    correct_valid = {k: 0 for k in ks}

    # Iterate over the validation set.
    for imgs, targets, masks in valid_loader:
        with torch.no_grad():
            # Move batch data to GPU.
            imgs = imgs.cuda()
            targets, masks = targets.squeeze(0).cuda(), masks.squeeze(0).cuda()

            # Forward pass.
            acts = forward_Feat(args, model, imgs)
            if torch.sum(masks) > 0:
                acts *= masks
            preds = forward_Exp(args, model, acts)

            # Calculate loss.
            loss = loss_fn(preds, targets, embeddings, train_label_indices)

            # Update loss.
            loss_valid += loss.data.detach().item()

            # Compute the cosine similarity between each prediction and
            # each ground-truth category embedding. Then sort the results
            # in descending order, so that the top `k` indices represent
            # the categories that are most similar to the model's prediction.
            cat_preds_per_sample = torch.argsort(
                (preds @ embeddings)
                / (vector_norm(preds, dim=1, keepdim=True) @
                   vector_norm(embeddings, dim=0, keepdim=True)),
                dim=1,
                descending=True
            )[:, :max(ks)]

            # Calculate accuracy.
            for k in ks:
                for img_idx, cat_preds in enumerate(cat_preds_per_sample):
                    correct_valid[k] += targets[img_idx, cat_preds[:k]] \
                        .any().detach().item()

        # Free memory.
        torch.cuda.empty_cache()

    # Calculate average loss and average accuracy per sample.
    loss_valid /= len(valid_loader)
    accs_valid = {k: correct_valid[k] / len(valid_loader.dataset) * 100
                  for k in ks}
    print()
    print(f"Valid average loss: {loss_valid:.6f}")
    for k in ks:
        print(f"Valid top-{k} accuracy: {accs_valid[k]:.2f}%")

    return loss_valid, accs_valid[1]


def main(args: argparse.Namespace):
    """
    Train a model on the specified dataset.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Make sure the run is deterministic and reproducible.
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up run name.
    if not args.name:
        args.name = f"vsf_{args.refer}_{args.model}_{args.layer}_" \
            f"{args.anno_rate:.1f}"
    if args.random:
        args.name += "_random"

    # Set up output path.
    args.save_dir = os.path.join(args.save_dir, args.name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set up wandb logging.
    if args.wandb:
        wandb_id_file_path = pathlib.Path(os.path.join(args.save_dir,
                                                       "runid.txt"))
        print("Creating new wandb instance...", wandb_id_file_path)
        run = wandb.init(project="temporal_scale",
                         name=args.name, config=args)
        wandb_id_file_path.write_text(str(run.id))

    # Set up GloVe word embeddings.
    glove = GloVe(name="6B", dim=args.word_embedding_dim)
    torch.cuda.empty_cache()

    # Set up dataset.
    if args.refer == "vg":
        # Set up Visual Genome dataset.
        root = os.path.join(args.data_dir, "vg")
        dataset = VisualGenomeInstances(
            root=root,
            transform=data_transforms["val"]
        )
        train_size = int(len(dataset) * args.train_rate)
        test_size = len(dataset) - train_size
        datasets = {}
        datasets["train"], datasets["val"] = \
            random_split(dataset, [train_size, test_size])
        label_indices = [glove.stoi[label]
                         for label in dataset.labels]
        embeddings = glove.vectors[label_indices].T.cuda()
        train_label_indices = np.random.choice(
            range(len(label_indices)),
            int(len(label_indices) * args.anno_rate)
        )
    elif args.refer == "coco":
        # Set up COCO dataset.
        root = os.path.join(args.data_dir, "coco")
        datasets = {}
        datasets["train"] = CocoInstances(
            root=os.path.join(root, "train2017"),
            annFile=os.path.join(root, "annotations/instances_train2017.json"),
            cat_mappings_file=os.path.join(root, "coco_label_embedding.pth"),
            transform=data_transforms["train"]
        )
        datasets["val"] = CocoInstances(
            root=os.path.join(root, "val2017"),
            annFile=os.path.join(root, "annotations/instances_val2017.json"),
            cat_mappings_file=os.path.join(root, "coco_label_embedding.pth"),
            transform=data_transforms["val"]
        )
        label_indices = list(datasets["train"].cat_mappings["itos"].keys())
        embeddings = glove.vectors[label_indices].T.cuda()
        train_label_indices = None
    else:
        raise NotImplementedError(f"Reference dataset '{args.refer}' is not "
                                  "implemented.")

    # Set up dataloader.
    dataloaders = {
        dataset_type: DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
        for dataset_type, dataset in datasets.items()
    }

    # Set up model.
    model = setup_explainer(args, random_feature=args.random)
    if args.model_path is None:
        args.model_path = os.path.join(args.save_dir, "ckpt_best.pth.tar")
    if os.path.exists(args.model_path):
        print(f"Loading model from '{args.model_path}'...")
        model.load_state_dict(torch.load(args.model_path)["state_dict"])
    else:
        print(f"No model found at '{args.model_path}'. "
              "Training from scratch...")
    model = model.cuda()

    # Set up optimizer, scheduler and loss function.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    loss_fn = CSMRLoss(margin=args.margin)

    # Print confirmation message.
    print()
    print("[ Model was set up successfully! ]".center(79, "-"))
    print()

    # ------------------------------ TRAINING ------------------------------- #

    loss_valid_best = 99999999
    accs_train = []
    accs_valid = []

    with open(os.path.join(args.save_dir, "valid.txt"), "w") as f:
        for epoch in range(args.epochs):
            print()
            print(f"[ Epoch {epoch} starting ]".center(79, "-"))

            # Train and validate.
            loss_train, acc_train = train_one_epoch(args,
                                                    model,
                                                    loss_fn,
                                                    dataloaders["train"],
                                                    embeddings,
                                                    epoch,
                                                    optimizer,
                                                    train_label_indices)
            loss_valid, acc_valid = validate(args,
                                             model,
                                             loss_fn,
                                             dataloaders["val"],
                                             embeddings,
                                             train_label_indices)

            # Setup wandb logging.
            if args.wandb:
                wandb.log({"Epoch": epoch})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Train_Loss": loss_train})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Train_Acc": acc_train})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Valid_Loss": loss_valid})
                wandb.log({"Epoch": epoch, "Epoch_Ave_Valid_Acc": acc_valid})
                wandb.log({"Epoch": epoch,
                           "LR": optimizer.param_groups[0]["lr"]})

            # Save train and validation accuracy.
            accs_train.append(acc_train)
            accs_valid.append(acc_valid)
            scheduler.step(loss_valid)
            f.write("epoch: %d\n" % epoch)
            f.write("train loss: %f\n" % loss_train)
            f.write("train accuracy: %f\n" % acc_train)
            f.write("valid loss: %f\n" % loss_valid)
            f.write("valid accuracy: %f\n" % acc_valid)

            # Save checkpoint if validation loss is the lowest loss so far.
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                print("==> new checkpoint saved")
                f.write("==> new checkpoint saved\n")
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    os.path.join(args.save_dir, "ckpt_best.pth.tar")
                )
                plt.figure()
                plt.plot(loss_train, "-o", label="train")
                plt.plot(loss_valid, "-o", label="valid")
                plt.xlabel("Epoch")
                plt.ylabel("Loss (")
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(args.save_dir, "losses.png"))
                plt.close()

    # Save wandb summary.
    wandb.run.summary["best_validation_loss"] = loss_valid_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-rate", type=float, default=0.1,
                        help="Fraction of concepts used for supervision")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for data loading")
    parser.add_argument("--classifier-name", type=str, default="fc",
                        help="Name of classifier layer")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--layer", type=str, default="layer4",
                        help="Target layer to explain")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Hyperparameter for margin ranking loss")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="Target network")
    parser.add_argument("--model-path", type=Optional[str], default=None,
                        help="Path to trained explainer model")
    parser.add_argument("--name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of subprocesses to use for data loading")
    parser.add_argument("--random", action="store_true",
                        help="Use a randomly initialized target model instead "
                        "of torchvision pretrained weights")
    parser.add_argument("--refer", type=str, default="coco",
                        choices=("vg", "coco"),
                        help="Reference dataset")
    parser.add_argument("--save-dir", type=str, default="./outputs",
                        help="Path to model checkpoints")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="How often to save a model checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed to use")
    parser.add_argument("--train-rate", type=float, default=0.9,
                        help="Fraction of data used for training")
    parser.add_argument("--wandb", action="store_true",
                        help="Use wandb for logging")
    parser.add_argument("--word-embedding-dim", type=int, default=300,
                        help="GloVe word embedding dimension to use")

    args = parser.parse_args()
    print(args)

    main(args)
