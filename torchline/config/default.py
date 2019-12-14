from .config import CfgNode as CN

_C = CN()
_C.VERSION = 1


# ---------------------------------------------------------------------------- #
# input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)

# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.NAME = 'cifar10'
_C.DATASET.BATCH_SIZE = 16
_C.DATASET.DIR = './datasets/skin100_dataset/train'
_C.DATASET.TRAIN_LIST = './datasets/train_skin10.txt'
_C.DATASET.VALID_LIST = './datasets/valid_skin10.txt'
_C.DATASET.TEST_LIST = './datasets/test_skin10.txt'
_C.DATASET.IS_TRAIN = False # specify to load training or testing set


# ---------------------------------------------------------------------------- #
# transforms
# ---------------------------------------------------------------------------- #

_C.TRANSFORMS = CN() # image transforms
_C.TRANSFORMS.NAME = 'DefaultTransforms'


## transforms for tensor
_C.TRANSFORMS.TENSOR = CN()
# for skin100
_C.TRANSFORMS.TENSOR.NORMALIZATION = CN()
_C.TRANSFORMS.TENSOR.NORMALIZATION.mean = [0.6075, 0.4564, 0.4182] 
_C.TRANSFORMS.TENSOR.NORMALIZATION.std = [0.2158, 0.1871, 0.1826]
# _C.TRANSFORMS.TENSOR.NORMALIZATION = {
#             'mean':[0.6054, 0.4433, 0.4084], 
#             'std': [0.2125, 0.1816, 0.1786]  # for skin10
_C.TRANSFORMS.TENSOR.RANDOM_ERASING = CN()
_C.TRANSFORMS.TENSOR.RANDOM_ERASING.enable = 0
_C.TRANSFORMS.TENSOR.RANDOM_ERASING.p = 0.5
_C.TRANSFORMS.TENSOR.RANDOM_ERASING.scale = (0.02, 0.3) # range of proportion of erased area against input image.
_C.TRANSFORMS.TENSOR.RANDOM_ERASING.ratio = (0.3, 3.3), # range of aspect ratio of erased area.


## transforms for PIL image
_C.TRANSFORMS.IMG = CN()

### modify the image size, only use one operation
# random_resized_crop
_C.TRANSFORMS.IMG.RANDOM_RESIZED_CROP = CN()
_C.TRANSFORMS.IMG.RANDOM_RESIZED_CROP.enable = 0
_C.TRANSFORMS.IMG.RANDOM_RESIZED_CROP.scale = (0.5, 1.0)
_C.TRANSFORMS.IMG.RANDOM_RESIZED_CROP.ratio = (3/4, 4/3)

# resize
_C.TRANSFORMS.IMG.RESIZE =  CN()
_C.TRANSFORMS.IMG.RESIZE.enable = 1

# random_crop
_C.TRANSFORMS.IMG.RANDOM_CROP = CN()
_C.TRANSFORMS.IMG.RANDOM_CROP.enable = 1

# center_crop
_C.TRANSFORMS.IMG.CENTER_CROP = CN()
_C.TRANSFORMS.IMG.CENTER_CROP.enable = 0

### without modifying the image size
_C.TRANSFORMS.IMG.AUG_IMAGENET = False
_C.TRANSFORMS.IMG.AUG_CIFAR = False

# color_jitter
_C.TRANSFORMS.IMG.COLOR_JITTER = CN()
_C.TRANSFORMS.IMG.COLOR_JITTER.enable = 0
_C.TRANSFORMS.IMG.COLOR_JITTER.brightness = 0.
_C.TRANSFORMS.IMG.COLOR_JITTER.contrast = 0.
_C.TRANSFORMS.IMG.COLOR_JITTER.saturation = 0.
_C.TRANSFORMS.IMG.COLOR_JITTER.hue = 0.

# horizontal_flip
_C.TRANSFORMS.IMG.RANDOM_HORIZONTAL_FLIP = CN()
_C.TRANSFORMS.IMG.RANDOM_HORIZONTAL_FLIP.enable = 1
_C.TRANSFORMS.IMG.RANDOM_HORIZONTAL_FLIP.p = 1

# vertical_flip
_C.TRANSFORMS.IMG.RANDOM_VERTICAL_FLIP = CN()
_C.TRANSFORMS.IMG.RANDOM_VERTICAL_FLIP.enable = 1
_C.TRANSFORMS.IMG.RANDOM_VERTICAL_FLIP.p = 1

# random_rotation
_C.TRANSFORMS.IMG.RANDOM_ROTATION = CN()
_C.TRANSFORMS.IMG.RANDOM_ROTATION.enable = 1
_C.TRANSFORMS.IMG.RANDOM_ROTATION.degrees = 15



_C.LABEL_TRANSFORMS = CN() # label transforms
_C.LABEL_TRANSFORMS.NAME = 'default'


# ---------------------------------------------------------------------------- #
# dataloader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.SAMPLER_TRAIN = "default"
_C.DATALOADER.SAMPLER_TEST = "default"


# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.META_ARCH = 'Resnet50'
_C.MODEL.WEIGHTS = ""
_C.MODEL.CLASSES = 10
_C.MODEL.PRETRAINED = True
_C.MODEL.FEATURES = ['f4', ]
_C.MODEL.FEATURES_FUSION = 'sum'


# ---------------------------------------------------------------------------- #
# optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()
_C.OPTIM.NAME = 'adam'
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.BASE_LR = 0.001
_C.OPTIM.WEIGHT_DECAY = 0.0001

# scheduler
_C.OPTIM.SCHEDULER = CN()
_C.OPTIM.SCHEDULER.NAME = 'CosineAnnealingLR'
_C.OPTIM.SCHEDULER.GAMMA = 0.1 # decay factor

# for CosineAnnealingLR
_C.OPTIM.SCHEDULER.T_MAX = 10 

# for ReduceLROnPlateau
_C.OPTIM.SCHEDULER.MODE = 'min' # min for loss, max for acc
_C.OPTIM.SCHEDULER.PATIENCE = 10
_C.OPTIM.SCHEDULER.VERBOSE = True # print log once update lr

# for StepLR
_C.OPTIM.SCHEDULER.STEP_SIZE = 10

# for MultiStepLR
_C.OPTIM.SCHEDULER.MILESTONES = [10, 15, 25, 35]

# ---------------------------------------------------------------------------- #
# loss
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()
_C.LOSS.NAME = 'CrossEntropy'

# ---------------------------------------------------------------------------- #
# hooks
# ---------------------------------------------------------------------------- #
_C.HOOKS = CN()

## EarlyStopping
_C.HOOKS.EARLY_STOPPING = CN()
_C.HOOKS.EARLY_STOPPING.type = 2 # 0: True 1: False 2: custom
_C.HOOKS.EARLY_STOPPING.monitor = 'val_loss'
_C.HOOKS.EARLY_STOPPING.min_delta = 0.
_C.HOOKS.EARLY_STOPPING.patience = 10
_C.HOOKS.EARLY_STOPPING.mode = 'min'
_C.HOOKS.EARLY_STOPPING.verbose = True

# ModelCheckpoint
_C.HOOKS.MODEL_CHECKPOINT = CN()
_C.HOOKS.MODEL_CHECKPOINT.type = 0 # 0: True 1: False 2: custom
_C.HOOKS.MODEL_CHECKPOINT.filepath = './output/checkpoints'
_C.HOOKS.MODEL_CHECKPOINT.monitor = 'val_loss'
_C.HOOKS.MODEL_CHECKPOINT.mode = 'min'


# ---------------------------------------------------------------------------- #
# Module template 
# ---------------------------------------------------------------------------- #

_C.MODULE_TEMPLATE = CN()
_C.MODULE_TEMPLATE.NAME = 'LightningTemplateModel'

# ---------------------------------------------------------------------------- #
# Trainer 
# ---------------------------------------------------------------------------- #

_C.TRAINER = CN()
_C.TRAINER.ACCUMULATE_GRAD_BATCHES = 1
_C.TRAINER.MIN_EPOCHS = 30
_C.TRAINER.MAX_EPOCHS = 1000
_C.TRAINER.GRAD_CLIP_VAL = 0.5 # clip gradient of which norm is larger than 0.5
_C.TRAINER.SHOW_PROGRESS_BAR = True # show progree bar
_C.TRAINER.ROW_LOG_INTERVAL = 100 # Every k batches lightning will make an entry in the metrics log
_C.TRAINER.LOG_SAVE_INTERVAL = 100 # Every k batches, lightning will write the new logs to disk, ie: save a .csv log file every 100 batches
_C.TRAINER.DEFAULT_SAVE_PATH = './output'
_C.TRAINER.LOG_GPU_MEMORY = "min_max" # 'min_max': log only the min/max utilization
_C.TRAINER.FAST_DEV_RUN = False # everything only with 1 training and 1 validation batch.
_C.TRAINER.LOGGER = CN()
_C.TRAINER.LOGGER.SETTING = 0 # 0: True  1: False  2: custom
_C.TRAINER.LOGGER.type = 'mlflow'
_C.TRAINER.LOGGER.MLFLOW = CN()
_C.TRAINER.LOGGER.MLFLOW.experiment_name = 'torchline_logs'
_C.TRAINER.LOGGER.MLFLOW.tracking_uri = _C.TRAINER.DEFAULT_SAVE_PATH
_C.TRAINER.LOGGER.TEST_TUBE = CN()
_C.TRAINER.LOGGER.TEST_TUBE.name = 'torchline_logs'
_C.TRAINER.LOGGER.TEST_TUBE.save_dir = _C.TRAINER.DEFAULT_SAVE_PATH
_C.TRAINER.LOGGER.TEST_TUBE.debug = False


# ---------------------------------------------------------------------------- #
# log
# ---------------------------------------------------------------------------- #

_C.LOG = CN()
_C.LOG.NAME = 'log.txt'

# ---------------------------------------------------------------------------- #
# Misc 
# ---------------------------------------------------------------------------- #

_C.SEED = 666
_C.DEFAULT_CUDNN_BENCHMARK = True
_C.TEST_ONLY = CN()
_C.TEST_ONLY.type = 'ckpt'
_C.TEST_ONLY.checkpoint_path = '' # load_from_checkpoint
_C.TEST_ONLY.weights_path = '' # load_from_metrics
_C.TEST_ONLY.tags_csv = ''
_C.TEST_ONLY.on_gpu = True
_C.TEST_ONLY.map_location = 'cuda:0'
_C.TEST_ONLY.test_file_path = '' # specify the path of images

