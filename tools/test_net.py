import sys
import logging

from seglossbias.utils import mkdir, setup_logging
from seglossbias.engine import default_argument_parser, load_config, DefaultTester

logger = logging.getLogger(__name__)


def setup(args):
    cfg = load_config(args)
    mkdir(cfg.OUTPUT_DIR)
    setup_logging(output_dir=cfg.OUTPUT_DIR)
    return cfg


def main():
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    tester = DefaultTester(cfg)
    tester.test()


if __name__ == "__main__":
    main()
