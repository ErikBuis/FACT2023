import argparse
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision


def setup_explainer(
    settings: argparse.Namespace,
    hook_fn: Optional[Callable] = None,
    random_feature: bool = False
    ) -> torch.nn.Module:
    """ This function is used to setup the explainer model.

    Args:
        settings (argparse.Namespace): The settings of the experiment.
        hook_fn (Optional[Callable], optional): The hook function to be registered to the explainer model. Defaults to None.
        random_feature (bool, optional): Whether to use random feature. Defaults to False.

    Raises:
        NotImplementedError: If the model is not supported.

    Returns:
        torch.nn.Module: The explainer model.
    """
    if random_feature:
        model = torchvision.models.__dict__[settings.model](pretrained=False) # Load the model without pretrained weights
    else:
        model = torchvision.models.__dict__[settings.model](pretrained=True) # Load the model with pretrained weights

    # Freeze the weights of the model
    for param in model.parameters():
        param.requires_grad = False

    # Get the index of the target layer (e.g., layer4)
    target_index = list(model._modules).index(settings.layer) 
    
    # Get the index of the classifier layer (e.g., fc)
    classifier_index = list(model._modules).index(settings.classifier_name) 
    
    # Get the feature dimension of the target layer (the last convolutional layer)
    # This only works for ResNet50. For other models, you need to change the code.
    # feature_dim = list(model._modules[settings.layer]._modules.values())[-1].conv3.out_channels
    # -> This was added by the improved code author, but it doesn't work for ResNet18 
    # However, this code makes more sense taking into consideration the paper's description of the model
    # "i.e. the output of the last convolutional layer with d filters."
   
    # Replace the layers after the target layer with identity layers
    # This is to avoid the computation of the layers after the target layer
    for module_name in list(model._modules)[target_index + 1:classifier_index]:
        if module_name[-4:] == 'pool':
            continue
        else:
            model._modules[module_name] = nn.Identity()

    if settings.model[:6] == 'resnet':
        # Get the feature dimension of the classifier layer
        feature_dim = model._modules[settings.classifier_name].in_features 
        
        # Replace the classifier layer with a new classifier layer with the same feature dimension
        model._modules[settings.classifier_name] = nn.Sequential( 
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.1),
            nn.Linear(
                in_features=feature_dim,
                out_features=feature_dim,
                bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.1),
            nn.Linear(in_features=feature_dim,
                      out_features=settings.word_embedding_dim,
                      bias=True))
    else:
        raise NotImplementedError

    # Load the pretrained weights of the explainer model if specified
    # This code was previously under the first if statement (if random_feature:)
    if settings.pretrain: 
        checkpoint = torch.load(settings.pretrain)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)

    # Register the hook function to the explainer model
    # The hook function will be called after the forward pass of the target layer
    if hook_fn: 
        model._modules.get(settings.layer).register_forward_hook(hook_fn)

    # Move the explainer model to GPU
    model.cuda() 
    
    return model

