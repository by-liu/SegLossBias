import numpy as np
import logging
from typing import Optional
from terminaltables import AsciiTable
from medpy import metric

from .evaluator import DatasetEvaluator
from .metric import dice_coef, getHausdorff

logger = logging.getLogger(__name__)


class RetinalLesionEvaluator(DatasetEvaluator):
    classes = ["MA", "iHE", "HaEx", "CWS", "vHE", "pHE", "NV", "FiP"]
    """Evaluate performance on Lesion segmentation by Dice coef"""
    def __init__(self, thres: float = 0.5) -> None:
        self.thres = thres
        self.all_dices: Optional[np.array] = None  # [N, C, X]
        # self.all_hd = []
        self.compute_hd95 = False

    def set_hd95(self):
        self.compute_hd95 = True
        self.all_hd95 = []

    def num_samples(self):
        return self.all_dices.shape[0] if self.all_dices is not None else 0

    def reset(self):
        self.all_dices = None

    def update_hd95(self, pred: np.array, target: np.array) -> np.array:
        pred = (pred > self.thres).astype(np.int8)
        for i in range(pred.shape[0]):
            result = pred[i]
            gt = target[i]
            if 0 == np.count_nonzero(result) or 0 == np.count_nonzero(gt):
                continue
            hd95_val = metric.hd95(result, gt)
            self.all_hd95.append(hd95_val)

    def mean_hd95(self):
        hd95_mean = np.mean(np.array(self.all_hd95))
        return hd95_mean

    def update(self, pred: np.array, target: np.array) -> np.array:
        pred = (pred > self.thres).astype(np.float32)
        # pred = (pred > 0.2).astype(np.float32)
        if target.ndim == 3:
            target = np.expand_dims(target, axis=1)

        dices = dice_coef(pred, target)

        if self.all_dices is None:
            self.all_dices = dices
        else:
            self.all_dices = np.concatenate((self.all_dices, dices), axis=0)

        return dices.mean()

    def main_metric(self):
        return "DSC"

    def mean_score(self):
        dice = np.mean(self.all_dices)
        return dice

    def class_score(self):
        if self.all_dices.shape[1] != len(self.classes):
            logger.warn(
                "The number of classes doesn't match {} vs. {}".format(self.all_dices.shape[1], len(self.classes))
            )
            return
        scores = np.mean(self.all_dices, axis=0)
        class_table_data = [["id"] + ["Class"] + ["DSC"]]
        for i in range(scores.shape[0]):
            class_table_data.append(
                [i] + [self.classes[i]] + ["{:.4f}".format(scores[i])]
            )
        class_table_data.append(
            [""] + ["mean"] + ["{:.4f}".format(np.mean(scores))]
        )
        table = AsciiTable(class_table_data)
        logger.info("\n" + table.table)
        return scores
