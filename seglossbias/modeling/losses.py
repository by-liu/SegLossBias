import logging
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from seglossbias.config.registry import Registry
from .compound_losses import (
    CompoundLoss, CrossEntropyWithL1, CrossEntropyWithKL, CrossEntropyWithDice, CrossEntropyWithDB,
    FocalWithDice, FocalWithKL, FocalWithL1
)
from .dice import DiceLoss, GeneralisedDiceLoss

logger = logging.getLogger(__name__)


LOSS_REGISTRY = Registry("loss")


@LOSS_REGISTRY.register("CE")
def build_ce(cfg):
    if cfg.MODEL.MODE == "multiclass":
        return nn.CrossEntropyLoss(
            weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None,
            ignore_index=cfg.LOSS.IGNORE_INDEX
        )
    else:
        return nn.BCEWithLogitsLoss()


LOSS_REGISTRY.register(
    "CE+L1",
    lambda cfg: CrossEntropyWithL1(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)


LOSS_REGISTRY.register(
    "CE+KL",
    lambda cfg: CrossEntropyWithKL(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)


LOSS_REGISTRY.register(
    "CE+DB",
    lambda cfg: CrossEntropyWithDB(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)


LOSS_REGISTRY.register(
    "CE+Dice",
    lambda cfg: CrossEntropyWithDice(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        log_dice=False,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)


LOSS_REGISTRY.register(
    "CE+LogDice",
    lambda cfg: CrossEntropyWithDice(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)

LOSS_REGISTRY.register(
    "Focal+Dice",
    lambda cfg: FocalWithDice(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        log_dice=False,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)

LOSS_REGISTRY.register(
    "Focal+LogDice",
    lambda cfg: FocalWithDice(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        log_dice=False,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)

LOSS_REGISTRY.register(
    "Focal+KL",
    lambda cfg: FocalWithKL(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)

LOSS_REGISTRY.register(
    "Focal+L1",
    lambda cfg: FocalWithL1(
        mode=cfg.MODEL.MODE,
        alpha=cfg.LOSS.ALPHA,
        factor=cfg.LOSS.ALPHA_FACTOR,
        step_size=cfg.LOSS.ALPHA_STEP_SIZE,
        temp=cfg.LOSS.TEMP,
        ignore_index=cfg.LOSS.IGNORE_INDEX,
        background_index=cfg.LOSS.BACKGROUND_INDEX,
        weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHTS) if cfg.LOSS.CLASS_WEIGHTS else None
    )
)

LOSS_REGISTRY.register(
    "DiceLoss",
    lambda cfg: DiceLoss(
        mode=cfg.MODEL.MODE,
        log_loss=False,
        ignore_index=cfg.LOSS.IGNORE_INDEX
    )
)


LOSS_REGISTRY.register(
    "LogDiceLoss",
    lambda cfg: DiceLoss(
        mode=cfg.MODEL.MODE,
        log_loss=True,
        ignore_index=cfg.LOSS.IGNORE_INDEX
    )
)

LOSS_REGISTRY.register(
    "GDiceLoss",
    lambda cfg : GeneralisedDiceLoss()
)

LOSS_REGISTRY.register(
    "FocalLoss",
    lambda cfg : smp.losses.FocalLoss(
        mode=cfg.MODEL.MODE,
        ignore_index=cfg.LOSS.IGNORE_INDEX
    )
)


def get_loss_func(cfg) -> nn.Module:
    """get the loss function given the loss name"""
    loss_name = cfg.LOSS.NAME
    loss_func = LOSS_REGISTRY.get(loss_name)(cfg)
    setattr(loss_func, 'name', loss_name)
    if isinstance(loss_func, CompoundLoss):
        num_terms = 3
    else:
        num_terms = 1
    setattr(loss_func, 'num_terms', num_terms)

    logger.info("Successfully build loss func : {}".format(loss_func))

    return loss_func
