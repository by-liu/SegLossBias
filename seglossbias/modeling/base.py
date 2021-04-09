import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from config.registry import Registry

_EPS = 1e-10

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        #return summary(self, input_shape=(2, 3, 224, 224))


class TwoItemsLoss(nn.Module):
    """
    The base class for implementing a two-item loss:
        l = l_1 + alpha * l_2
    """
    def __init__(self, alpha : float = 0.1,
                 factor : float = 1,
                 step_size : int = 0,
                 max_alpha : float = 100.) -> None:
        super().__init__()
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size

    def adjust_alpha(self, epoch : int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            self.alpha = min(self.alpha * self.factor, self.max_alpha)


def kl_div(p : torch.Tensor, q : torch.Tensor) -> torch.Tensor:
    x = p * torch.log(p / q)
    return x.abs().mean()
    # return x.mean()


def get_region_size(x : torch.Tensor, valid_mask=None) -> torch.Tensor:
    if valid_mask is not None:
        x = torch.einsum("bcwh,bwh->bcwh", x, valid_mask)
        cardinality = torch.einsum("bwh->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3]

    region_size = (
        (torch.einsum("bcwh->bc", x) + _EPS) / (cardinality + _EPS)
    )

    # if x.ndim == 4:
    #     region_size = (
    #         (torch.einsum("bcwh->bc", x) + _EPS)
    #         / (x.shape[2] * x.shape[3])
    #     )
    # else:
    #     region_size = (
    #         (torch.einsum("cwh->c", x) + _EPS)
    #         / (x.shape[1] * x.shape[2])
    #     )
    return region_size


class KlLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs : torch.Tensor, targets : torch.Tensor):
        loss_kl = kl_div(targets, inputs)
        return loss_kl


class WeightedBCE(nn.Module):
    """
    WCE as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        N = probs.size(0)
        probs = probs.view(N, -1)
        targets = targets.view(N, -1)
        # import ipdb; ipdb.set_trace()
        # print((targets.sum(-1) + _EPS))
        # print(((1 - targets).sum(-1) + _EPS))

        loss = - (
            (1 / (targets.sum(-1) + _EPS)) * (targets * torch.log(probs + _EPS)).sum(-1)
            + (1 / ((1 - targets).sum(-1) + _EPS)) * ((1. - targets) * torch.log(1 - probs + _EPS)).sum(-1)
        )
        # print(loss)

        return loss.mean()
