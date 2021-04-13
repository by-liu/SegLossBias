import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import os
import random
from datetime import datetime
import torch

from .file_io import mkdir

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = None, deterministic: bool = False):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
        deterministic (bool):  Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(output_dir : str = "", level=logging.INFO, stream : bool = True) -> None:
    formatter = logging.Formatter(
        fmt='%(asctime)s,%(msecs)03d %(levelname)s '
            '[%(filename)s:%(lineno)s - %(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(level)
    if stream or not output_dir:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
    if output_dir:
        mkdir(output_dir)
        date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        handler = RotatingFileHandler(
            os.path.join(output_dir, "log.{}".format(date_time)),
            encoding="utf-8",
            maxBytes=100 * 1024 * 1024,  # 100M
            backupCount=100
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    logger.info("Setting up logger complete.")


def get_logfile(logger):
    if len(logger.root.handlers) == 1:
        return None
    else:
        return logger.root.handlers[1].baseFilename
