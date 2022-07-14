"""
File: defaults.py
Author: Binguan Liu
Date: Dec 14, 2020
Brief: default configurations for segmentation
"""

from yacs.config import CfgNode as CN


_C = CN()

# -------------------------------------------------------------
# Model options
# -------------------------------------------------------------
_C.MODEL = CN()

# Model architecture
_C.MODEL.ARCH = "unet"

# Model architecture
_C.MODEL.ENCODER = "resnet50"

# Pretrained weights for encoder net
_C.MODEL.ENCODER_WEIGHTS = ""

# activation function
_C.MODEL.ACT_FUNC = "softmax"

# The path of pretrained model
_C.MODEL.PRETRAINED = ""

# The number of classes to predict
_C.MODEL.NUM_CLASSES = 8

# Mode : binary, multiclass or multilabel
_C.MODEL.MODE = "binary"

# The number of input channels
_C.MODEL.INPUT_CHANNELS = 3

# Rate of dropout
_C.MODEL.DROPOUT = -1.0

# -------------------------------------------------------------
# Dataset options
# -------------------------------------------------------------
_C.DATA = CN()

_C.DATA.NAME = "retinal-lesions"

# The root directory of dataset
_C.DATA.DATA_ROOT = ""

# The label values of pixels in the mask
_C.DATA.LABEL_VALUES = [255]

# Available for retinal-lesions dataset.
# If true, convert the data setting to binary classification
_C.DATA.BINARY = False

# If true, get and output the region size as well
_C.DATA.REGION_SIZE = False

# If true, get and output the region size normalized by the area as well
_C.DATA.NORMALIZE_REGION_SIZE = False

# If true, get and output the region number as well
_C.DATA.REGION_NUMBER = False

# The mean value of the raw pixels across the R G B channels.
_C.DATA.MEAN = [0.485, 0.456, 0.406]

# The std value of the raw pixels across the R G B channels.
_C.DATA.STD = [0.229, 0.224, 0.225]

# The target size of image resize
_C.DATA.RESIZE = [512, 512]

# How many subprocesses to use for data loading.
_C.DATA.NUM_WORKERS = 8

# For retianl-lesion-class setting
_C.DATA.CLASS_NAME = "hard_exudate"

# For poly dataset setting
_C.DATA.SET_NAME = "Kvasir"

# -------------------------------------------------------------
# Optimizer options
# -------------------------------------------------------------
_C.LOSS = CN()

# Name of loss function
_C.LOSS.NAME = "bce_logit"

# Hyper parameter of loss
_C.LOSS.ALPHA = 0.1

# Step size of adjusting hyper weight
# If zero, it won't change the weight during the training
_C.LOSS.ALPHA_STEP_SIZE = 0

# Factor of increasing hyper weight when it triggers adjusting
_C.LOSS.ALPHA_FACTOR = 5.0

# temperature
_C.LOSS.TEMP = 1.0

# Label smoothing for soft bce loss
_C.LOSS.LABEL_SMOOTHING = 0.1

# The target value that is igored and does not contribute to param optimization
_C.LOSS.IGNORE_INDEX = 255

# For some losses, the background index is required
_C.LOSS.BACKGROUND_INDEX = -1

# Weighing different classes if necessary
_C.LOSS.CLASS_WEIGHTS = []

# -------------------------------------------------------------
# Optimizer options
# -------------------------------------------------------------
_C.SOLVER = CN()

# Optimization method
_C.SOLVER.OPTIMIZING_METHOD = "adam"

# Base learning rate
_C.SOLVER.BASE_LR = 0.1

# minimum learning rate (trigged for some schedulers)
_C.SOLVER.MIN_LR = 0.0

# Learning rate policy
_C.SOLVER.LR_POLICY = "reduce_on_plateau"

# Available for ReduceLROnPlateau
_C.SOLVER.FACTOR = 0.5
_C.SOLVER.PATIENCE = 3
_C.SOLVER.REDUCE_MODE = "min"

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 10

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Number of warmup epochs.
_C.SOLVER.WARMUP_EPOCH = 0

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# -------------------------------------------------------------
# Training options
# -------------------------------------------------------------
_C.TRAIN = CN()

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 4

# train data list path (A relative path to _C.DATA.DATA_ROOT or an absoulte path)
_C.TRAIN.DATA_PATH = ""

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Save model checkpoint after period epochs.
_C.TRAIN.CHECKPOINT_AFTER_PERIOD = 5

# If True, caculate metric (auc/F1/dice/...) in training phase.
# May be very costy due to the large size of traing samples
_C.TRAIN.CALCULATE_METRIC = False

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = False

# -------------------------------------------------------------
# Validation options
# -------------------------------------------------------------
_C.VAL = CN()

# Total mini-batch size.
_C.VAL.BATCH_SIZE = 4

# Val data list path (A relative path to _C.DATA.DATA_ROOT or an absoulte path)
_C.VAL.DATA_PATH = ""

# -------------------------------------------------------------
# Test options (Only available when running test script)
# -------------------------------------------------------------
_C.TEST = CN()

# Val data list path (A relative path to _C.DATA.DATA_ROOT or an absoulte path)
_C.TEST.DATA_PATH = ""

# The split name
_C.TEST.SPLIT = "test"

# Total mini-batch size.
_C.TEST.BATCH_SIZE = 4

# The path of the testing checkpoint file.
# If empty, it will load model indicated in the best_checkpoint file
_C.TEST.CHECKPOINT_PATH = ""

# If True, it will load model indicated in the best_checkpoint file
_C.TEST.BEST_CHECKPOINT = False

# The model to be tested indexed by epoch (start from 1)
_C.TEST.MODEL_EPOCH = 0

# If True, it will save the predicted results into one numpy array file
_C.TEST.SAVE_PREDICTS = False

# -------------------------------------------------------------
# Transductive options (Only available when running transductive script)
# -------------------------------------------------------------
_C.TRANSDUCTIVE = CN()

# If true, employ ground-truth labels
_C.TRANSDUCTIVE.USE_LABEL = True

# Hyper parameter of loss
_C.TRANSDUCTIVE.ALPHA = 1.0

# Optimization method
_C.TRANSDUCTIVE.OPTIMIZING_METHOD = "sgd"

# Base learning rate
_C.TRANSDUCTIVE.BASE_LR = 0.1

# Base learning rate
_C.TRANSDUCTIVE.MAX_ITER = 50

# Momentum.
_C.TRANSDUCTIVE.MOMENTUM = 0.9

# Momentum dampening.
_C.TRANSDUCTIVE.DAMPENING = 0.0

# Nesterov momentum.
_C.TRANSDUCTIVE.NESTEROV = True

# Exponential decay factor.
_C.TRANSDUCTIVE.GAMMA = 0.1

# L2 regularization.
_C.TRANSDUCTIVE.WEIGHT_DECAY = 1e-4

_C.TRANSDUCTIVE.VERBOSE = False


# -------------------------------------------------------------
# Wandb(https://wandb.ai/) : Experiment management platform
# -------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.PROJECT = "iccv2021"
_C.WANDB.ENTITY = "newton"


# -------------------------------------------------------------
# Misc options
# -------------------------------------------------------------
# Output basedir.
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.RNG_SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False

# LOG preriod in iters
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# The device name
_C.DEVICE = "cuda:0"

# If True, perform test after training
_C.PERFORM_TEST = False

# Threshold for determining the positive segmentaiton results
_C.THRES = 0.5


def get_cfg() -> CN:
    """
    Get a copy of the default configuration.
    """
    cfg = _C.clone()
    return cfg
