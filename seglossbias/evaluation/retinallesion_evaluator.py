import numpy as np
import logging
from typing import Optional
from terminaltables import AsciiTable
from medpy import metric
import surface_distance
import wandb

from .evaluator import DatasetEvaluator
from .metric import dice_coef

logger = logging.getLogger(__name__)


class RetinalLesionEvaluator(DatasetEvaluator):
    classes = ["MA", "iHE", "HaEx", "CWS", "vHE", "pHE", "NV", "FiP"]
    """Evaluate performance on Lesion segmentation by Dice coef"""
    def __init__(self, thres: float = 0.5) -> None:
        self.thres = thres
        self.all_dices: Optional[np.array] = None  # [N, C, X]
        # self.all_hd = []
        self.compute_hd95 = False
        self.compute_nsd = False

    def set_hd95(self):
        self.compute_hd95 = True
        self.all_hd95 = []

    def set_nsd(self):
        self.compute_nsd = True
        self.all_nsd = []

    def num_samples(self):
        return self.all_dices.shape[0] if self.all_dices is not None else 0

    def reset(self):
        self.all_dices = None
        self.all_hd95 = []
        self.all_nsd = []

    def update_nsd(self, pred: np.array, target: np.array) -> np.array:
        pred = (pred > self.thres).astype(np.int8)
        for i in range(pred.shape[0]):
            result = np.squeeze(pred[i])
            gt = np.squeeze(target[i])
            if 0 == np.count_nonzero(gt):
                continue
            distances = surface_distance.compute_surface_distances(
                gt.astype(np.bool), result.astype(np.bool), spacing_mm=(2, 1)
            )
            nsd = surface_distance.compute_surface_dice_at_tolerance(
                distances, tolerance_mm=1
            )

            self.all_nsd.append(nsd)

    def mean_nsd(self):
        nsd_mean = np.mean(np.array(self.all_nsd))
        return nsd_mean

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

        self.curr = {"dsc": float(dices.mean())}

        if self.compute_hd95:
            self.update_hd95(pred, target)

        if self.compute_nsd:
            self.update_nsd(pred, target)

        return self.curr["dsc"]

    def main_metric(self):
        return "dsc"

    def curr_score(self):
        return self.curr

    def mean_score(self, all_metric=False):
        dice = float(np.mean(self.all_dices))

        scores = np.mean(self.all_dices, axis=0)
        class_table_data = [["id"] + ["Class"] + ["DSC"]]
        for i in range(scores.shape[0]):
            class_table_data.append(
                [i] + [self.classes[i]] + ["{:.4f}".format(scores[i])]
            )
        class_table_data.append(
            [None] + ["mean"] + ["{:.4f}".format(np.mean(scores))]
        )

        metric = {"dsc": dice}
        if self.compute_hd95:
            hd = self.mean_hd95()
            metric["hd95"] = float(hd)
        
        if self.compute_nsd:
            nsd = self.mean_nsd()
            metric["nsd"] = float(nsd)

        if not all_metric:
            return dice
        else:
            return metric, class_table_data

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
            [None] + ["mean"] + ["{:.4f}".format(np.mean(scores))]
        )
        table = AsciiTable(class_table_data)
        logger.info("\n" + table.table)
        return scores

    def wandb_score_table(self):
        _, table_data = self.mean_score(all_metric=True)
        return wandb.Table(
            columns=table_data[0],
            data=table_data[1:]
        )