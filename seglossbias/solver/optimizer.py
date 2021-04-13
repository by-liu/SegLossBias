import torch
from yacs.config import CfgNode as CN

from .lr_scheduler import Poly


def build_optimizer(
    cfg : CN,
    model : torch.nn.Module
) -> torch.optim.Optimizer:
    optim_params = model.parameters()

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def build_scheduler(
    cfg : CN,
    optimizer : torch.optim.Optimizer,
    data_loader : torch.utils.data.DataLoader
) -> torch.optim.lr_scheduler:
    if cfg.SOLVER.LR_POLICY == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.MAX_EPOCH,
            eta_min=cfg.SOLVER.MIN_LR
        )
    elif cfg.SOLVER.LR_POLICY == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.SOLVER.STEP_SIZE,
            gamma=cfg.SOLVER.GAMMA,
        )
    elif cfg.SOLVER.LR_POLICY == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.SOLVER.REDUCE_MODE,
            factor=cfg.SOLVER.FACTOR,
            patience=cfg.SOLVER.PATIENCE,
            verbose=True
        )
    elif cfg.SOLVER.LR_POLICY == "poly":
        scheduler = Poly(
            optimizer,
            num_epochs=cfg.SOLVER.MAX_EPOCH,
            iters_per_epoch=len(data_loader),
            min_lr=cfg.SOLVER.MIN_LR
        )
    else:
        raise NotImplementedError(
            "Does not support {} scheduler".format(cfg.SOLVER.LR_POLICY)
        )

    setattr(scheduler, "name", cfg.SOLVER.LR_POLICY)

    return scheduler


def get_lr(optimizer : torch.optim.Optimizer) -> float:
    for param in optimizer.param_groups:
        return param["lr"]
