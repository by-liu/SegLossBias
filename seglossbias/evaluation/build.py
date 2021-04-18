from seglossbias.config.registry import Registry
from .retinallesion_evaluator import RetinalLesionEvaluator
from .citycapes_evaluator import CityscapesEvaluator


EVALUATOR_REGISTRY = Registry("evaluator")
EVALUATOR_REGISTRY.register(
    "retinal-lesions",
    lambda cfg: RetinalLesionEvaluator(thres=cfg.THRES)
)
EVALUATOR_REGISTRY.register(
    "cityscapes",
    lambda cfg: CityscapesEvaluator()
)


def build_evaluator(cfg):
    return EVALUATOR_REGISTRY.get(cfg.DATA.NAME)(cfg)
