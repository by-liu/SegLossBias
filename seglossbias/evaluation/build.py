from seglossbias.config.registry import Registry
from .retinallesion_evaluator import RetinalLesionEvaluator
from .citycapes_evaluator import CityscapesEvaluator
from .segment_evaluator import SegmentEvaluator


EVALUATOR_REGISTRY = Registry("evaluator")
EVALUATOR_REGISTRY.register(
    "retinal-lesions",
    lambda cfg: RetinalLesionEvaluator(thres=cfg.THRES)
)
EVALUATOR_REGISTRY.register(
    "cityscapes",
    lambda cfg: CityscapesEvaluator()
)
EVALUATOR_REGISTRY.register(
    "polyp",
    lambda cfg: SegmentEvaluator(
        classes=["_background_", "polyp"],
        ignore_index=cfg.LOSS.IGNORE_INDEX,
    )
)


def build_evaluator(cfg):
    return EVALUATOR_REGISTRY.get(cfg.DATA.NAME)(cfg)
