import sys
import logging

from seglossbias.utils import mkdir, set_random_seed, setup_logging
from seglossbias.engine import default_argument_parser, load_config, DefaultTrainer

logger = logging.getLogger(__name__)


def setup(args):
    cfg = load_config(args)
    mkdir(cfg.OUTPUT_DIR)
    setup_logging(output_dir=cfg.OUTPUT_DIR)
    set_random_seed(
        seed=None if cfg.RNG_SEED < 0 else cfg.RNG_SEED,
        deterministic=False if cfg.RNG_SEED < 0 else True
    )
    return cfg


def main():
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    trainer = DefaultTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
