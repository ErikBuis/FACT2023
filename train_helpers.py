from typing import Optional

import torch
import torch.nn as nn

bgr_mean = [109.5388, 118.6897, 124.6901]


def set_bn_eval(m: nn.Module):
    """ Set the BatchNorm layer to eval mode

    Args:
        m (nn.Module): The module to be set to eval mode.
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
    if classname.find('BatchNorm1d') != -1:
        m.eval()


class Identity(nn.Module):
    """ Identity layer

    Args:
        nn (torch.nn.Module): The identity layer.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x) -> torch.Tensor:
        return x


class CSMRLoss(torch.nn.Module):
    """Cosine Similarity Margin Ranking Loss

    Shape:
        - output: :math:`(N, D)` where `N` is the batch size and `D` is the feature dimension.
    """

    def __init__(self, margin=1):
        super(CSMRLoss, self).__init__()
        self.margin = torch.tensor(margin).cuda()

    def forward(
        self, 
        output: torch.Tensor,
        target_onehot: torch.Tensor,
        embeddings: torch.Tensor,
        train_label_idx: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

        # Compute the cosine similarity between the output and the embeddings
        cosine_similarity = torch.mm(output, embeddings) / \
                            torch.mm(torch.sqrt(torch.sum(output ** 2, dim=1, keepdim=True)),
                                     torch.sqrt(torch.sum(embeddings ** 2, dim=0, keepdim=True)))
        
        # If train_label_idx is not None, only compute the loss for the training labels       
        if train_label_idx:
            target_onehot = target_onehot[:, train_label_idx]
            cosine_similarity = cosine_similarity[:, train_label_idx]
            indices = torch.sum(target_onehot, dim=1) > 0
            cosine_similarity = cosine_similarity[indices]
            target_onehot = target_onehot[indices]
            
        false_terms = (1 - target_onehot) * cosine_similarity # Get false terms cosine similarity
        tmp = torch.sum(target_onehot * cosine_similarity, dim=1) / torch.sum(target_onehot, dim=1) # Normalized target cosine similarity?
        loss = (1 - target_onehot) * (self.margin - tmp.unsqueeze(1) + false_terms)

        # Set the loss to 0 if the cosine similarity is greater than the margin
        loss[torch.isnan(loss)] = 0.
        loss = torch.max(torch.tensor(0.).cuda(), loss.float())
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        
        return loss
