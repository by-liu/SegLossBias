import numpy as np
import logging
from typing import Optional
from terminaltables import AsciiTable

from .evaluator import DatasetEvaluator
from .metric import dice_coef

logger = logging.getLogger(__name__)


class RetinalLesionEvaluator(DatasetEvaluator):
    classes = ["MA", "iHE", "HaEx", "CWS", "vHE", "pHE", "NV", "FiP"]
    """Evaluate performance on Lesion segmentation by Dice coef"""
    def __init__(self, thres: float = 0.5) -> None:
        self.thres = thres
        self.all_dices: Optional[np.array] = None  # [N, C, X]

    def num_samples(self):
        return self.all_dices.shape[0] if self.all_dices is not None else 0

    def reset(self):
        self.all_dices = None

    def update(self, pred: np.array, target: np.array) -> np.array:
        pred = (pred > self.thres).astype(np.float32)
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
        return np.mean(self.all_dices)

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
