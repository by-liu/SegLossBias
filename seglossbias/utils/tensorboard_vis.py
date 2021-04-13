
import logging
import numpy as np
import torch
import os.path as osp
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode as CN
from typing import Optional, Union, List
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class TensorboardWriter:
    """
    Helper class to log information to Tensorboard
    """

    def __init__(self, cfg : CN) -> None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.log_dir = osp.join(cfg.OUTPUT_DIR, "runs.{}".format(date_time))
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.classes = None
        self.plot_class_score = cfg.TENSORBOARD.PLOT_CLASS_SCORE

        logger.info("To see logged results in Tensorboard, please launch using the command:")
        logger.info("tensorboard --logdir {}".format(self.log_dir))

    def add_scalars(self, data_dict : dict, global_step : Optional[int] = None) -> None:
        if self.writer is not None:
            for key, item in data_dict.items():
                self.writer.add_scalar(key, item, global_step)

    def add_class_score(self, tag : str,
                        scores : Union[np.ndarray, torch.tensor],
                        global_step : Optional[int] = None) -> None:
        cell_text : List[str] = []
        for val in scores:
            cell_text.append(["{:.2f}".format(val * 100)])
        cell_text.append(["{:.2f}".format(scores.mean() * 100)])

        row_labels = self.classes if self.classes else list(map(int, range(val.shape[0])))
        row_labels = row_labels + ["mean"]
        col_labels = ["Score (%)"]

        # fig, ax = plt.subplots()
        fig = plt.Figure(facecolor="w", edgecolor="k")
        ax = fig.add_subplot()

        # hide axies
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")
        ax.table(cellText=cell_text,
                 rowLabels=row_labels,
                 colLabels=col_labels,
                 rowLoc="center",
                 cellLoc="center",
                 loc="center")

        fig.set_tight_layout(True)

        if self.writer is not None:
            self.writer.add_figure(
                tag=tag,
                figure=fig,
                global_step=global_step
            )

    def flush(self):
        self.writer.flush()

    def close(self):
        self.flush()
        self.writer.close()
