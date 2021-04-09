import os.path as osp
import logging
import torch
from yacs.config import CfgNode as CN
from typing import Optional

from .file_io import mkdir, load_list

logger = logging.getLogger(__name__)


def save_checkpoint(
    save_dir : str,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : torch.optim.lr_scheduler,
    epoch : int,
    last_checkpoint : bool = True,
    best_checkpoint : bool = False,
    val_score : Optional[float] = None,
) -> None:
    mkdir(save_dir)
    model_name = "checkpoint_epoch_{}.pth".format(epoch + 1)
    model_path = osp.join(save_dir, model_name)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    if val_score:
        state["val_score"] = val_score
    torch.save(state, model_path)
    if last_checkpoint:
        with open(osp.join(save_dir, "last_checkpoint"), "w") as wf:
            wf.write(model_name)
    if best_checkpoint:
        with open(osp.join(save_dir, "best_checkpoint"), "w") as wf:
            wf.write(model_name)


def load_checkpoint(model_path : str, model : torch.nn.Module, device) -> None:
    if not osp.exists(model_path):
        raise FileNotFoundError(
            "Model not found : {}".format(model_path)
        )
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    logger.info("Succeed to load weights from {}".format(model_path))
    if missing_keys:
        logger.warn("Missing keys : {}".format(missing_keys))
    if unexpected_keys:
        logger.warn("Unexpected keys : {}".format(unexpected_keys))


def get_best_model_path(cfg : CN) -> str:
    """get the path of the best model"""
    best_checkpoint_path = osp.join(cfg.OUTPUT_DIR, "model", "best_checkpoint")
    if not osp.exists(best_checkpoint_path):
        raise FileNotFoundError(
            "File not found : {}".format(best_checkpoint_path)
        )
    model_name = load_list(best_checkpoint_path)[0]

    return osp.join(cfg.OUTPUT_DIR, "model", model_name)


def get_last_model_path(cfg : CN) -> str:
    """get the path of the best model"""
    last_checkpoint_path = osp.join(cfg.OUTPUT_DIR, "model", "last_checkpoint")
    if not osp.exists(last_checkpoint_path):
        raise FileNotFoundError(
            "File not found : {}".format(last_checkpoint_path)
        )
    model_name = load_list(last_checkpoint_path)[0]

    return osp.join(cfg.OUTPUT_DIR, "model", model_name)


def load_train_checkpoint(cfg : CN, model : torch.nn.Module,
                          optimizer : torch.optim.Optimizer = None,
                          scheduler : torch.optim.lr_scheduler = None) -> int:
    if not cfg.TRAIN.AUTO_RESUME:
        return 0, -1, None

    try:
        last_checkpoint_path = get_last_model_path(cfg)
        checkpoint = torch.load(last_checkpoint_path, map_location=cfg.DEVICE)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("Succeed to load weights from {}".format(last_checkpoint_path))
        best_checkpoint_path = get_best_model_path(cfg)
        checkpoint = torch.load(best_checkpoint_path, map_location=cfg.DEVICE)
        best_epoch = checkpoint["epoch"]
        best_score = checkpoint["val_score"] if "val_score" in checkpoint else None
        return epoch + 1, best_epoch, best_score
    except Exception:
        return 0, -1, None
