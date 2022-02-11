import os.path as osp
import time
import pprint
import logging
import torch
from yacs.config import CfgNode as CN
import wandb

from seglossbias.config import convert_cfg_to_dict
from seglossbias.modeling import build_model, get_loss_func, CompoundLoss
from seglossbias.solver import build_optimizer, build_scheduler, get_lr
from seglossbias.evaluation import build_evaluator, AverageMeter, LossMeter
from seglossbias.data import build_data_pipeline
from seglossbias.utils import (
    load_train_checkpoint, save_checkpoint, load_checkpoint
)

logger = logging.getLogger(__name__)


class DefaultTrainer:
    """
    A trainer with default training logic. It does the following:

    1. Create a model, loss, optimizer, LR scheduler
    2. build dataloader to generate mini-batch input data
    3. Create loss meter and performance evaluator
    4. Start training and evaluation in epoch-by-epoch manner
    5. Status logging, Tensorboard writing [optional] or wandb tracking [optional]
    """
    def __init__(self, cfg: CN):
        self.cfg = cfg
        logger.info("DefaultTrainer with config : ")
        logger.info(pprint.pformat(cfg))
        self.device = torch.device(cfg.DEVICE)
        self.build_model()
        self.build_dataloader()
        self.build_solver()
        self.build_meter()
        self.init_wandb_or_not()

    def build_model(self):
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        self.loss_func = get_loss_func(self.cfg)
        self.loss_func.to(self.device)

    def build_dataloader(self):
        self.train_loader = build_data_pipeline(self.cfg, "train")
        self.val_loader = build_data_pipeline(self.cfg, "val")

    def build_solver(self):
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.scheduler = build_scheduler(self.cfg, self.optimizer, self.train_loader)

    def build_meter(self):
        self.evaluator = build_evaluator(self.cfg)
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = LossMeter(self.loss_func.num_terms)

    def reset_meter(self):
        self.evaluator.reset()
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()

    def init_wandb_or_not(self):
        if self.cfg.WANDB.ENABLE:
            wandb.init(
                project=self.cfg.WANDB.PROJECT,
                entity=self.cfg.WANDB.ENTITY,
                config=convert_cfg_to_dict(self.cfg),
                tags=["train"]
            )
            wandb.run.name = "{}-{}-{}".format(
                wandb.run.id, self.cfg.DATA.NAME, self.cfg.LOSS.NAME
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def start_or_resume(self):
        self.start_epoch, self.best_epoch, self.best_score = load_train_checkpoint(
            self.cfg, self.model, self.optimizer, self.scheduler)

    def wandb_iter_info_or_not(
        self, iter, max_iter, epoch, phase="train",
        loss_meter=None, score=None, lr=None
    ):
        if not self.cfg.WANDB.ENABLE:
            return

        step = epoch * max_iter + iter
        log_dict = {"iter": step}

        if loss_meter is not None:
            loss_dict = loss_meter.get_vals()
            for key, val in loss_dict.items():
                log_dict["{}/Iter/{}".format(phase, key)] = val
        if score is not None:
            log_dict["{}/Iter/{}".format(phase, self.evaluator.main_metric())] = score
        if lr is not None:
            log_dict["{}/Iter/lr".format(phase)] = lr
        wandb.log(log_dict)

    def log_iter_info(
        self, iter, max_iter, epoch, phase="train",
        data_time_meter=None, batch_time_meter=None,
        loss_meter=None, score=None, lr=None
    ):
        log_str = []
        log_str.append("{} Epoch[{}][{}/{}]".format(phase, epoch + 1, iter + 1, max_iter))
        if data_time_meter is not None:
            log_str.append(
                "Data {data_time.val:.3f} ({data_time.avg:.3f})".format(data_time=data_time_meter)
            )
        if batch_time_meter is not None:
            log_str.append(
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})".format(batch_time=batch_time_meter)
            )
        if loss_meter is not None:
            log_str.append(loss_meter.print_status())
        if score is not None:
            log_str.append("{} {:.4f}".format(self.evaluator.main_metric(), score))
        if lr is not None:
            log_str.append("LR {:.3g}".format(lr))
        logger.info("\t".join(log_str))

    def tensorbaord_epoch_info_or_not(
        self, epoch, phase="train", evaluator=None,
        loss_meter=None
    ):
        if self.writer is None:
            return
        if loss_meter is not None:
            loss_dict = loss_meter.get_avgs()
            for key, val in loss_dict.items():
                self.writer.add_scalars(
                    {"{}/Epoch/{}".format(phase, key): val},
                    global_step=epoch
                )
        if isinstance(self.loss_func, CompoundLoss):
            self.writer.add_scalars(
                {"{}/Epoch/alpha".format(phase): self.loss_func.alpha}
            )
        if evaluator is not None:
            self.writer.add_scalars(
                {"{}/Epoch/{}".format(phase, evaluator.main_metric()): evaluator.mean_score()},
                global_step=epoch
            )

    def wandb_epoch_info_or_not(
        self, epoch, phase="train", evaluator=None,
        loss_meter=None
    ):
        if not self.cfg.WANDB.ENABLE:
            return
        if phase != "test":
            log_dict = {"epoch": epoch}
        else:
            log_dict = {"test_epoch": epoch}
        if loss_meter is not None:
            loss_dict = loss_meter.get_vals()
            for key, val in loss_dict.items():
                log_dict["{}/Epoch/{}".format(phase, key)] = val
        if isinstance(self.loss_func, CompoundLoss):
            log_dict["{}/Epoch/alpha".format(phase)] = self.loss_func.alpha
        if evaluator is not None:
            log_dict["{}/Epoch/{}".format(phase, evaluator.main_metric())] = evaluator.mean_score()
        wandb.log(log_dict)

    def log_epoch_info(
        self, epoch, phase="train", evaluator=None,
        loss_meter=None
    ):
        log_str = []
        log_str.append("{} Epoch[{}]".format(phase, epoch + 1))
        if evaluator is not None:
            log_str.append("Samples[{}]".format(evaluator.num_samples()))
        if loss_meter is not None:
            log_str.append(loss_meter.print_avg())
        if isinstance(self.loss_func, CompoundLoss):
            log_str.append("alpha {:.4f}".format(self.loss_func.alpha))
        if evaluator is not None:
            log_str.append("{} {:.4f}".format(evaluator.main_metric(), evaluator.mean_score()))
        logger.info("\t".join(log_str))

        if phase != "train" and self.cfg.MODEL.NUM_CLASSES > 1:
            evaluator.class_score()

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)
        lr = get_lr(self.optimizer)

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
                self.log_iter_info(i, max_iter, epoch,
                                   phase="train",
                                   data_time_meter=self.data_time_meter,
                                   batch_time_meter=self.batch_time_meter,
                                   loss_meter=self.loss_meter,
                                   score=score, lr=lr)
                self.wandb_iter_info_or_not(
                    i, max_iter, epoch,
                    phase="train",
                    loss_meter=self.loss_meter,
                    score=score,
                    lr=lr
                )
            end = time.time()
        self.log_epoch_info(epoch,
                            phase="train",
                            evaluator=self.evaluator,
                            loss_meter=self.loss_meter)
        self.tensorbaord_epoch_info_or_not(
            epoch, phase="train",
            evaluator=self.evaluator,
            loss_meter=self.loss_meter
        )
        self.wandb_epoch_info_or_not(
            epoch, phase="train",
            evaluator=self.evaluator,
            loss_meter=self.loss_meter
        )
        logger.info("====== Complete training epoch {} ======".format(epoch + 1))

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="val"):
        self.reset_meter()

        self.model.eval()

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
                self.log_iter_info(i, max_iter, epoch,
                                   phase=phase,
                                   data_time_meter=self.data_time_meter,
                                   batch_time_meter=self.batch_time_meter,
                                   loss_meter=self.loss_meter,
                                   score=score)
                self.wandb_iter_info_or_not(
                    i, max_iter, epoch, phase=phase, loss_meter=self.loss_meter, score=score
                )
            end = time.time()
        self.log_epoch_info(epoch,
                            phase=phase,
                            evaluator=self.evaluator,
                            loss_meter=self.loss_meter)
        self.tensorbaord_epoch_info_or_not(
            epoch, phase=phase, evaluator=self.evaluator,
            loss_meter=self.loss_meter
        )
        self.wandb_epoch_info_or_not(
            epoch, phase=phase, evaluator=self.evaluator,
            loss_meter=self.loss_meter
        )

        return self.loss_meter.avg(0), self.evaluator.mean_score()

    def val_epoch_or_not(self, epoch):
        if (epoch + 1) % self.cfg.TRAIN.EVAL_PERIOD == 0 or\
                (epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch)
            if self.scheduler.name == "reduce_on_plateau":
                self.scheduler.step(val_loss if self.scheduler.mode == "min" else val_score)
            return val_loss, val_score
        else:
            return None, None

    def test_epoch_or_not(self):
        if self.cfg.PERFORM_TEST:
            logger.info("Start testing ... ")
            epoch = self.best_epoch if self.cfg.TEST.BEST_CHECKPOINT else self.cfg.SOLVER.MAX_EPOCH - 1
            model_path = osp.join(
                self.cfg.OUTPUT_DIR, "model", "checkpoint_epoch_{}.pth".format(epoch + 1)
            )
            load_checkpoint(model_path, self.model, self.device)
            test_loader = build_data_pipeline(self.cfg, "test")
            test_loss, test_score = self.eval_epoch(
                test_loader, epoch, phase="test"
            )
            logger.info("Complete testing !")
            logger.info(
                ("Final performance on test subset- "
                 "model epoch {}, score {:.4f}").format(epoch, test_score)
            )

    def wandb_best_model_or_not(self):
        if self.cfg.WANDB.ENABLE:
            epoch = self.best_epoch if self.cfg.TEST.BEST_CHECKPOINT else self.cfg.SOLVER.MAX_EPOCH
            model_path = osp.join(
                self.cfg.OUTPUT_DIR, "model", "checkpoint_epoch_{}.pth".format(epoch)
            )
            wandb.save(model_path)

    def save_checkpoint_or_not(self, epoch, val_score):
        if (epoch + 1) >= self.cfg.TRAIN.CHECKPOINT_AFTER_PERIOD and\
                (epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            save_checkpoint(
                osp.join(self.cfg.OUTPUT_DIR, "model"),
                self.model, self.optimizer, self.scheduler, epoch,
                last_checkpoint=True,
                best_checkpoint=val_score > self.best_score if self.best_score is not None else True, 
                val_score=val_score
            )

            if self.best_score is None or val_score > self.best_score:
                self.best_score, self.best_epoch = val_score, epoch

    def train(self):
        self.start_or_resume()
        # Perform the training loop
        logger.info("Start training ... ")
        for epoch in range(self.start_epoch, self.cfg.SOLVER.MAX_EPOCH):
            # train phase
            self.train_epoch(epoch)
            val_loss, val_score = self.val_epoch_or_not(epoch)
            self.save_checkpoint_or_not(epoch, val_score)
            if self.scheduler.name not in {"reduce_on_plateau", "poly"}:
                self.scheduler.step()
            if isinstance(self.loss_func, CompoundLoss):
                self.loss_func.adjust_alpha(epoch)

        logger.info("Complete training !")
        logger.info(
            ("Best performance on validation subset - "
             "model epoch {}, score {:.4f}").format(self.best_epoch + 1, self.best_score)
        )
        # Peform test phase if requried
        self.test_epoch_or_not()
        self.wandb_best_model_or_not()
        self.close()
