import numpy as np
import torch
import logging
from typing import Optional, Union, List

from ..utils.constants import EPS

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val : float, n : int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeter:
    """A class wrapper to record the values of a loss function"""
    def __init__(self, num_terms: int = 1, names: Optional[List] = None) -> None:
        self.num_terms = num_terms
        self.names = (
            names if names is not None
            else ["loss" if i == 0 else "loss_" + str(i) for i in range(self.num_terms)]
        )
        self.meters = [AverageMeter() for _ in range(self.num_terms)]

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def avg(self, index=None):
        if index is None:
            ret = {}
            for name, meter in zip(self.names, self.meters):
                ret[name] = meter.avg
            return ret
        else:
            return self.meters[index].avg

    def update(self, val, n: int = 1):
        if not isinstance(val, tuple):
            val = [val]
        for x, meter in zip(val, self.meters):
            if type(x) == torch.Tensor:
                x = x.item()
            meter.update(x, n)

    def get_vals(self):
        ret = {}
        for name, meter in zip(self.names, self.meters):
            ret[name] = meter.val
        return ret

    def print_status(self):
        ret = []
        for name, meter in zip(self.names, self.meters):
            ret.append("{} {:.4f} ({:.4f})".format(name, meter.val, meter.avg))
        return "\t".join(ret)

    def get_avgs(self):
        ret = {}
        for name, meter in zip(self.names, self.meters):
            ret[name] = meter.avg
        return ret

    def print_avg(self):
        ret = []
        for name, meter in zip(self.names, self.meters):
            ret.append("{} {:.4f}".format(name, meter.avg))
        return "\t".join(ret)


def dice_coef(pred: Union[torch.Tensor, np.array],
              target : Union[torch.Tensor, np.array]) -> Union[torch.Tensor, np.array]:
    assert pred.shape == target.shape, "The shapes of input and target do not match"

    einsum = torch.einsum if type(pred) == torch.Tensor else np.einsum

    inter = einsum("ncij,ncij->nc", pred, target)
    union = einsum("ncij->nc", pred) + einsum("ncij->nc", target)

    dice = (2 * inter + EPS) / (union + EPS)

    return dice


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1)
    )
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1)
    )
    area_label, _ = np.histogram(
        label, bins=np.arange(num_classes + 1)
    )
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label
