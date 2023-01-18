import argparse
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision


def setup_explainer(args: argparse.Namespace,
                    hook_fn: Optional[Callable] = None,
                    random_feature: bool = False) -> nn.Module:
    """
    This function is used to set up the explainer model.

    Args:
        settings (argparse.Namespace): The settings of the experiment.
        hook_fn (Optional[Callable], optional): The hook function to be
            registered to the explainer model. Defaults to None.
        random_feature (bool, optional): Whether to use randomly initialized
            models instead of pretrained feature extractors. Defaults to False.

    Returns:
        nn.Module: The explainer model.
    """
    if random_feature:
        # Load the model without pretrained weights.
        model = torchvision.models.__dict__[args.model](weights=None)
    else:
        # Load the model with pretrained weights.
        # The argument "pretrained" is depricated, but there is no way to
        # specify the "weights" argument since the args.model string does not
        # correspond in general to the upper- and lowercase letters used in
        # torchvision.models. Therefore, we ignore the warning thrown here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = torchvision.models.__dict__[args.model](pretrained=True)

    # Freeze the weights of the target model.
    for param in model.parameters():
        param.requires_grad = False

    # Get the index of the target layer (e.g. layer4).
    target_index = list(model._modules).index(args.layer)

    # Get the index of the classifier layer (e.g. fc).
    classifier_index = list(model._modules).index(args.classifier_name)

    # Replace the layers between the target and classifier layer with identity
    # layers. This is done to avoid unnecessary computation of these layers.
    for module_name in list(model._modules)[target_index + 1:classifier_index]:
        if not module_name.endswith("pool"):
            model._modules[module_name] = nn.Identity()

    if args.model.startswith("resnet"):
        # Get the feature dimension of the classifier layer.
        feature_dim = model._modules[args.classifier_name].in_features

        # Replace the classifier layer with our Feature Explainer model.
        model._modules[args.classifier_name] = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.1),
            nn.Linear(in_features=feature_dim,
                      out_features=feature_dim,
                      bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.1),
            nn.Linear(in_features=feature_dim,
                      out_features=args.word_embedding_dim,
                      bias=True)
        )
    else:
        raise NotImplementedError(f"Model '{args.model}' is not supported.")

    # Load the pretrained weights of the explainer model if specified.
    # This code was previously under the first if-statement
    # (i.e. "if random_feature:").
    if args.pretrain:
        checkpoint = torch.load(args.pretrain)
        if type(checkpoint).__name__ == "OrderedDict" \
                or type(checkpoint).__name__ == "dict":
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict)

    # Register the hook function to the explainer model.
    # The hook function will be called every time  the forward pass through the
    # target layer has been performed. It is only intended for debugging/
    # profiling purposes.
    if hook_fn is not None:
        model._modules.get(args.layer).register_forward_hook(hook_fn)

    # Move the feature extracter model (Feat) and the explainer model (Exp),
    # which are now combined into `model`, to the GPU.
    model.cuda()

    return model
