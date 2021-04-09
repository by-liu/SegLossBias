from ..config.registry import Registry
from .retinallesion_evaluator import RetinalLesionEvaluator


EVALUATOR_REGISTRY = Registry("evaluator")

EVALUATOR_REGISTRY.register(
    "retinal-lesions",
    lambda cfg: RetinalLesionEvaluator(thres=cfg.THRES)
)


def build_evaluator(cfg):
    return EVALUATOR_REGISTRY.get(cfg.DATA.NAME)(cfg)
