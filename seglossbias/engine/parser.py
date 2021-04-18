import argparse
from yacs.config import CfgNode as CN

from ..config import get_cfg


def default_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Segmentation Pipeline')
    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using command-line")

    return parser


def load_config(args: argparse.Namespace) -> CN:
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    return cfg
