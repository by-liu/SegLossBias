import os.path as osp
from yacs.config import CfgNode as CN

from .tester import DefaultTester
from seglossbias.modeling import build_model


class TesterV2(DefaultTester):
    def __init__(self, cfg: CN):
        super().__init__(cfg)

    def build_model(self):
        if self.cfg.TEST.CHECKPOINT_PATH:
            model_path = self.cfg.TEST.CHECKPOINT_PATH
            # model_path = osp.join(
            #     self.cfg.OUTPUT_DIR,
            #     self.cfg.TEST.CHECKPOINT_PATH
            # )
        else:
            model_path = osp.join(
                self.cfg.OUTPUT_DIR, "best.pth"
            )

        self.model = build_model(self.cfg, model_path=model_path)
        self.model.to(self.device)
