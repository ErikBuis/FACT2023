"""This file was added by improved_lavise"""

import numpy as np
import torch


def unnorm(img: torch.Tensor,
           mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406]),
           std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225])) \
        -> torch.Tensor:
    """
    Unnormalize a tensor image with mean and standard deviation.

    Args:
        img (torch.Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (torch.Tensor): Mean for each channel.
            Default: ImageNet (?) mean.
        std (torch.Tensor): Standard deviation for each channel.
            Default: ImageNet (?) std.

    Returns:
        torch.Tensor: Unnormalized image.
    """
    img = img.permute(0, 2, 3, 1)
    img = img * std
    img = img + mean
    return img.permute(0, 3, 1, 2)


def combine_heatmap_img(img: torch.Tensor, activation: torch.Tensor,
                        heatmap_opacity: float = 0.60) -> torch.Tensor:
    """
    Combine an image and an activation map to create a heatmap.

    Args:
        img (torch.Tensor): Image tensor of size (C, H, W).
        activation (torch.Tensor): Activation tensor of size (H, W).
        heatmap_opacity (float): Opacity of the heatmap. Default: 0.60.

    Returns:
        torch.Tensor: Heatmap image tensor of size (C, H, W).
    """
    activation = activation/activation.max()
    activation = np.repeat(np.expand_dims(activation, 0), 3, axis=0)
    heatmap_img = heatmap_opacity * activation + (1-heatmap_opacity) * img
    return heatmap_img
