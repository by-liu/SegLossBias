import numpy as np
import logging
from typing import Optional
from terminaltables import AsciiTable
from medpy import metric
from typing import List
import wandb

from .evaluator import DatasetEvaluator
from .metric import dice_coef, intersect_and_union
from seglossbias.utils.constants import EPS

logger = logging.getLogger(__name__)

class SegmentEvaluator(DatasetEvaluator):
    def __init__(self,
                 classes: Optional[List[str]] = None,
                 ignore_index: int = 255) -> None:
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.ignore_index = ignore_index

    def num_samples(self):
        return self.nsamples

    def reset(self):
        self.total_area_inter = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_union = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_pred = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_target = np.zeros((self.num_classes, ), dtype=np.float)
        self.nsamples = 0

    def main_metric(self):
        return "mdsc"

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update all the metric from batch size prediction and target.

        Args:
            pred: predictions to be evaluated in one-hot formation
            y: ground truth. It should be one-hot format.
        """
        pred = np.argmax(pred, axis=1)
        assert pred.shape == target.shape, "pred and target should have same shapes"

        n = pred.shape[0]
        self.nsamples += n

        batch_area_inter = np.zeros((self.num_classes, ), dtype=np.float)
        batch_area_union = np.zeros((self.num_classes, ), dtype=np.float)
        batch_area_pred = np.zeros((self.num_classes, ), dtype=np.float)
        batch_area_target = np.zeros((self.num_classes, ), dtype=np.float)
        for i in range(n):
            area_inter, area_union, area_pred, area_target = (
                intersect_and_union(
                    pred[i], target[i], self.num_classes, self.ignore_index
                )
            )
            batch_area_inter += area_inter
            batch_area_union += area_union
            batch_area_pred += area_pred
            batch_area_target += area_target

        # update the total
        self.total_area_inter += batch_area_inter
        self.total_area_union += batch_area_union
        self.total_area_pred += batch_area_pred
        self.total_area_target += batch_area_target

        dice = 2 * batch_area_inter[1:].sum() / (batch_area_pred[1:].sum() + batch_area_target[1:].sum() + EPS)

        self.curr = {"dsc": dice}

        return dice

    def curr_score(self):
        return self.curr

    def mean_score(self, all_metric=False):
        # get per-class scores
        class_acc = self.total_area_inter[1:] / (self.total_area_target [1:]+ EPS)
        class_dice = (
            2 * self.total_area_inter[1:]
            / (self.total_area_pred[1:] + self.total_area_target[1:] + EPS)
        )
        class_iou = self.total_area_inter[1:] / (self.total_area_union[1:] + EPS)

        # get mean scores
        macc = np.mean(class_acc)
        miou = np.mean(class_iou)
        mdice = np.mean(class_dice)

        metric = {"mdsc": mdice, "macc": macc, "miou": miou}

        if not all_metric:
            return metric[self.main_metric()]


        columns = ["id", "class", "iou", "dsc", "acc"]
        class_table_data = [columns]
        for i in range(class_acc.shape[0]):
            class_table_data.append([
                i+1, self.classes[i+1],
                "{:.4f}".format(class_iou[i]),
                "{:.4f}".format(class_dice[i]),
                "{:.4f}".format(class_acc[i])
            ])
        class_table_data.append([
            None, "mean",
            "{:.4f}".format(miou),
            "{:.4f}".format(mdice),
            "{:.4f}".format(macc)
        ]) 

        return metric, class_table_data

    def wandb_score_table(self):
        _, table_data = self.mean_score(all_metric=True)
        return wandb.Table(
            columns=table_data[0],
            data=table_data[1:]
        )
