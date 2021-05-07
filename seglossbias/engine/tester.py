import os.path as osp
import time
import pprint
import logging
import torch
import cv2
import numpy as np
from yacs.config import CfgNode as CN

from seglossbias.modeling import build_model
from seglossbias.evaluation import build_evaluator, AverageMeter
from seglossbias.data import build_data_pipeline
from seglossbias.utils import (
    get_best_model_path, get_last_model_path, mkdir
)

logger = logging.getLogger(__name__)


class DefaultTester:
    """
    A tester with default testing logic. It does the following:

    1. Create a model and init with trained weights
    2. build dataloader to generate mini-batch input data
    3. Create loss meter and performance evaluator
    4. Loop through all the dataset to evaluate and save results if required
    """
    def __init__(self, cfg: CN):
        self.cfg = cfg
        logger.info("DefaultTester with config : ")
        logger.info(pprint.pformat(self.cfg))
        self.device = torch.device(self.cfg.DEVICE)
        self.data_loader = build_data_pipeline(self.cfg, self.cfg.TEST.SPLIT)
        self.build_model()
        if self.cfg.TEST.SAVE_PREDICTS:
            self.save_path = osp.join(self.cfg.OUTPUT_DIR,
                                      "{}_results".format(self.cfg.TEST.SPLIT))
            mkdir(self.save_path)
        self.build_meter()

    def build_model(self):
        if self.cfg.TEST.CHECKPOINT_PATH:
            model_path = self.cfg.TEST.CHECKPOINT_PATH
        elif self.cfg.TEST.MODEL_EPOCH > 0:
            model_path = osp.join(
                self.cfg.OUTPUT_DIR,
                "model/checkpoint_epoch_{}.pth".format(self.cfg.TEST.MODEL_EPOCH)
            )
        elif self.cfg.TEST.BEST_CHECKPOINT:
            model_path = get_best_model_path(self.cfg)
        else:
            model_path = get_last_model_path(self.cfg)

        self.model = build_model(self.cfg, model_path=model_path)
        self.model.to(self.device)

    def build_meter(self):
        self.evaluator = build_evaluator(self.cfg)
        self.batch_time_meter = AverageMeter()

    def reset_meter(self):
        self.evaluator.reset()
        self.batch_time_meter.reset()

    def log_iter_info(self, iter, max_iter, batch_time_meter=None, score=None):
        log_str = []
        log_str.append("Test Epoch[{}/{}]".format(iter + 1, max_iter))
        if batch_time_meter is not None:
            log_str.append(
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})"
                .format(batch_time=batch_time_meter)
            )
        if score is not None:
            log_str.append("{} {:.4f}".format(self.evaluator.main_metric(), score))
        logger.info("\t".join(log_str))

    def log_epoch_info(self, evaluator):
        log_str = []
        log_str.append("Test Samples[{}]".format(evaluator.num_samples()))
        log_str.append("{} {:.4f}".format(evaluator.main_metric(), evaluator.mean_score()))
        logger.info("\t".join(log_str))

        if self.cfg.MODEL.NUM_CLASSES > 1:
            evaluator.class_score()

    def save_predicts_or_not(self, predicts, sample_ids):
        if not self.cfg.TEST.SAVE_PREDICTS:
            return

        if self.cfg.MODEL.NUM_CLASSES == 1:
            pred_labels = (predicts.squeeze(dim=1) > self.cfg.THRES).int().cpu().numpy()
        else:
            pred_labels = torch.argmax(predicts, dim=1).cpu().numpy()

        for i, sample_id in enumerate(sample_ids):
            out_image = pred_labels[i].astype(np.uint8)
            out_file = osp.join(self.save_path, sample_id + ".png")
            cv2.imwrite(out_file, out_image)

    @torch.no_grad()
    def test(self):
        self.reset_meter()
        self.model.eval()

        max_iter = len(self.data_loader)
        end = time.time()
        for i, samples in enumerate(self.data_loader):
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.model(inputs)
            predicts = self.model.act(outputs)
            score = self.evaluator.update(predicts.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            self.save_predicts_or_not(predicts, samples[-1])
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter,
                                   batch_time_meter=self.batch_time_meter,
                                   score=score)

            end = time.time()
        self.log_epoch_info(self.evaluator)


class ImageFolderTester(DefaultTester):
    """
    A tester for inference with a given image folder as input,
    Compared with Default Tester, it doesn't contain the evaluation but save all the output masks.
    """
    def __init__(self, cfg: CN, save_path: str):
        self.cfg = cfg
        logger.info("ImageFolderTester with config : ")
        logger.info(pprint.pformat(self.cfg))
        self.device = torch.device(self.cfg.DEVICE)
        self.data_loader = build_data_pipeline(self.cfg, self.cfg.TEST.SPLIT)
        self.build_model()
        self.save_path = save_path
        mkdir(self.save_path)

    def save_predicts(self, predicts, sample_ids):
        if self.cfg.MODEL.NUM_CLASSES == 1:
            pred_labels = (predicts.squeeze(dim=1) > self.cfg.THRES).int().cpu().numpy()
        else:
            pred_labels = torch.argmax(predicts, dim=1).cpu().numpy()

        for i, sample_id in enumerate(sample_ids):
            out_image = np.uint8(pred_labels[i] * 255)
            out_file = osp.join(self.save_path, osp.splitext(sample_id)[0] + ".png")
            cv2.imwrite(out_file, out_image)

    @torch.no_grad()
    def test(self):
        timer = AverageMeter()

        self.model.eval()
        max_iter = len(self.data_loader)
        end = time.time()
        for i, samples in enumerate(self.data_loader):
            inputs, sample_ids = samples[0].to(self.device), samples[1]
            # forward
            outputs = self.model(inputs)
            predicts = self.model.act(outputs)
            # save predicts to predicted mask image
            self.save_predicts(predicts, sample_ids)
            timer.update(time.time() - end)
            logger.info(
                "Test Epoch[{}/{}] Time {timer.val:.3f} ({timer.avg:.3f})".format(
                    i + 1, max_iter, timer=timer)
            )
            end = time.time()
        logger.info("Done with test Samples[{}]".format(len(self.data_loader.dataset)))
