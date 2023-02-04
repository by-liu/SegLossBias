import logging
from seglossbias.data.build import build_data_pipeline
import time
from yacs.config import CfgNode as CN
import json
import wandb
import torch
import os.path as osp
from shutil import copyfile
from terminaltables.ascii_table import AsciiTable

from .trainer import DefaultTrainer
from seglossbias.solver import get_lr
from seglossbias.utils import round_dict, save_checkpoint_v2, load_checkpoint, load_train_checkpoint_v2
from seglossbias.modeling.compound_losses import CompoundLoss

logger = logging.getLogger(__name__)


class TrainerV2(DefaultTrainer):
    def __init__(self, cfg: CN):
        super().__init__(cfg)

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        logger.info("====== Start training epoch {} ======".format(epoch + 1))
        for i, samples in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
                # For compounding loss, make sure the first term represents the overall loss
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # Adjusting LR by iteration if poly is used
            if self.scheduler.name == "poly":
                self.scheduler.step(epoch=epoch)
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = self.model.act(outputs)
            score = self.evaluator.update(predicts.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_epoch_info(epoch)
        logger.info("====== Complete training epoch {} ======".format(epoch + 1))

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())
        logger.info(
            "{} Iter[{}/{}][{}]\t{}".format(
                phase, iter + 1, max_iter, epoch + 1, json.dumps(round_dict(log_dict))
            )
        )
        if self.cfg.WANDB.ENABLE and phase.lower() == "train":
            wandb_log_dict = {"iter": epoch * max_iter + iter}
            wandb_log_dict.update(dict(
                ("{}/Iter/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_epoch_info(self, epoch, phase="Train"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, CompoundLoss):
            log_dict["alpha"] = self.loss_func.alpha
        metric, table_data = self.evaluator.mean_score(all_metric=True)
        log_dict.update(metric)
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if phase.lower() != "train":
            logger.info("\n" + AsciiTable(table_data).table)
        if self.cfg.WANDB.ENABLE:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            if phase.lower() != "train":
                wandb_log_dict["{}/score_table".format(phase)] = wandb.Table(
                    columns=table_data[0], data=table_data[1:]
                )
            wandb.log(wandb_log_dict)

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="val"):
        self.reset_meter()
        self.model.eval()

        if phase.lower() == "test" and self.cfg.DATA.NAME == "retinal-lesions":
            # self.evaluator.compute_hd95 = True
            self.evaluator.compute_nsd = True

        max_iter = len(data_loader)
        end = time.time()
        for i, samples in enumerate(data_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            # metric
            self.loss_meter.update(loss)
            predicts = self.model.act(outputs)
            score = self.evaluator.update(predicts.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter, epoch, phase=phase)
            end = time.time()
        self.log_epoch_info(epoch, phase=phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score()

    def start_or_resume(self):
        if self.cfg.TRAIN.AUTO_RESUME:
            self.start_epoch, self.best_epoch, self.best_score = (
                load_train_checkpoint_v2(
                    self.cfg.OUTPUT_DIR, self.device, self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
            )
        else:
            self.start_epoch, self.best_epoch, self.best_score = 0, -1, None

    def train(self):
        self.start_or_resume()
        logger.info("Start training ... ")
        for epoch in range(self.start_epoch, self.cfg.SOLVER.MAX_EPOCH):
            self.train_epoch(epoch)
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch, phase="Val")
            if self.best_score is None or val_score > self.best_score:
                self.best_score, self.best_epoch = val_score, epoch
                best_checkpoint = True
            else:
                best_checkpoint = False
            if self.scheduler.name not in {"reduce_on_plateau", "poly"}:
                self.scheduler.step()
            elif self.scheduler.name == "reduce_on_plateau":
                self.scheduler.step(val_loss if self.scheduler.mode == "min" else val_score)
            if isinstance(self.loss_func, CompoundLoss):
                self.loss_func.adjust_alpha(epoch)
            save_checkpoint_v2(
                self.cfg.OUTPUT_DIR, self.model, self.optimizer, self.scheduler, epoch,
                best_checkpoint=best_checkpoint,
                val_score=val_score
            )
            logger.info(
                "Epoch[{}]\tBest {} on Val : {:.4f} at epoch {}".format(
                    epoch + 1, self.evaluator.main_metric(),
                    self.best_score, self.best_epoch + 1
                )
            )
            if self.cfg.WANDB.ENABLE and best_checkpoint:
                wandb.log({
                    "epoch": epoch,
                    "Val/best_epoch": self.best_epoch,
                    "Val/best_{}".format(self.evaluator.main_metric()): self.best_score,
                    "Val/best_score_table": self.evaluator.wandb_score_table()
                })
        if self.cfg.WANDB.ENABLE:
            copyfile(
                osp.join(self.cfg.OUTPUT_DIR, "best.pth"),
                osp.join(self.cfg.OUTPUT_DIR, "{}-best.pth".format(wandb.run.name))
            )
        logger.info("Complete training !")
        if self.cfg.PERFORM_TEST:
            self.test()

    # Peform test phase if requried
    def test(self):
        logger.info("We are almost done : final testing ...")
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.cfg.OUTPUT_DIR, "best.pth"), self.model, self.device
        )
        test_loader = build_data_pipeline(self.cfg, "test")
        self.eval_epoch(test_loader, epoch, phase="Test")
