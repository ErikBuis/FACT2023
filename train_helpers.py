from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_bn_eval(m: nn.Module):
    """
    Set the BatchNorm layer to eval mode.

    Args:
        m (nn.Module): The module to be set to eval mode.
    """
    classname = m.__class__.__name__
    if "BatchNorm2d" in classname or "BatchNorm1d" in classname:
        m.eval()


class Identity(nn.Module):
    """Identity layer. Performs the identity operation."""
    def __init__(self):
        """Set up an Identity layer instance."""
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the identity layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor.
        """
        return x


class CSMRLoss(nn.Module):
    """Cosine Similarity Margin Ranking Loss."""

    def __init__(self, margin: int = 1):
        """
        Set up a Cosine Similarity Margin Ranking Loss instance.

        Args:
            margin (int, optional): The margin to be used in the loss.
                Defaults to 1.
        """
        super(CSMRLoss, self).__init__()
        self.margin = torch.tensor(margin).cuda()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                embeddings: torch.Tensor,
                train_label_indices: Optional[np.ndarray] = None) \
            -> torch.Tensor:
        """
        Forward pass of the Cosine Similarity Margin Ranking Loss.
        This is an algorithmically efficient implementation of the mathematical
        loss formulation described in the paper.

        Args:
            preds (torch.Tensor): The predictions of the explainer model.
                Shape: [batch_size, embedding_dim].
            targets (torch.Tensor): The multiple-hot target category vector.
                Shape: [batch_size, num_categories].
            embeddings (torch.Tensor): The ground-truth category embeddings.
                Shape: [embedding_dim, num_categories]
            train_label_indices (Optional[np.ndarray], optional): The indices
                of the labels to train on. Defaults to None.
                Shape: [num_train_labels].

        Returns:
            torch.Tensor: The average loss per sample in the batch.
        """
        # Compute the cosine similarity between each prediction and each
        # ground-truth category embedding.
        cosine_similarity = (preds @ embeddings) \
            / (torch.sqrt(torch.sum(preds**2, dim=1, keepdim=True)) @
               torch.sqrt(torch.sum(embeddings**2, dim=0, keepdim=True)))

        # If train_label_idx is not None, only compute the loss for the
        # training labels. Mask any samples where a non-training label is the
        # ground-truth label.
        if train_label_indices is not None:
            targets = targets[:, train_label_indices]
            cosine_similarity = cosine_similarity[:, train_label_indices]
            indices = torch.any(targets, dim=1)
            cosine_similarity = cosine_similarity[indices]
            targets = targets[indices]

        # Calculate the similarity between the positive concept and the
        # prediction (i.e. dot preduct between v_t_j and v_hat) per sample.
        pos_concept_sim = torch.sum(targets * cosine_similarity, dim=1) \
            / torch.sum(targets, dim=1)

        # Calculate the similarity between each negative concept and the
        # prediction (i.e. dot products between v_c and v_hat for all c in C
        # where c =/= t_j) per sample.
        neg_concept_sim = (1 - targets) * cosine_similarity
        loss = (1 - targets) \
            * (self.margin - pos_concept_sim.unsqueeze(1) + neg_concept_sim)

        # Prevent NaNs (created via division by zero) from propagating.
        loss[torch.isnan(loss)] = 0

        # Calculate the final loss per sample and average over the batch.
        loss = torch.max(torch.tensor(0).cuda(), loss.float())
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        return loss
