from .config import CfgNode as CN

_C = CN()
_C.VERSION = 1


# ---------------------------------------------------------------------------- #
# input
# ---------------------------------------------------------------------------- #
_C.input = CN()
_C.input.size = (224, 224)

# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #
_C.dataset = CN()
_C.dataset.name = 'cifar10'
_C.dataset.batch_size = 16
_C.dataset.dir = './datasets/skin100_dataset/train'
_C.dataset.train_list = './datasets/train_skin10.txt'
_C.dataset.valid_list = './datasets/valid_skin10.txt'
_C.dataset.test_list = './datasets/test_skin10.txt'
_C.dataset.is_train = False # specify to load training or testing set


# ---------------------------------------------------------------------------- #
# transforms
# ---------------------------------------------------------------------------- #

_C.transforms = CN() # image transforms
_C.transforms.name = 'DefaultTransforms'


## transforms for tensor
_C.transforms.tensor = CN()
# for skin100
_C.transforms.tensor.normalization = CN()
_C.transforms.tensor.normalization.mean = [0.6075, 0.4564, 0.4182] 
_C.transforms.tensor.normalization.std = [0.2158, 0.1871, 0.1826]
# _C.transforms.tensor.normalization = {
#             'mean':[0.6054, 0.4433, 0.4084], 
#             'std': [0.2125, 0.1816, 0.1786]  # for skin10
_C.transforms.tensor.random_erasing = CN()
_C.transforms.tensor.random_erasing.enable = 0
_C.transforms.tensor.random_erasing.p = 0.5
_C.transforms.tensor.random_erasing.scale = (0.02, 0.3) # range of proportion of erased area against input image.
_C.transforms.tensor.random_erasing.ratio = (0.3, 3.3), # range of aspect ratio of erased area.


## transforms for PIL image
_C.transforms.img = CN()

### modify the image size, only use one operation
# random_resized_crop
_C.transforms.img.random_resized_crop = CN()
_C.transforms.img.random_resized_crop.enable = 0
_C.transforms.img.random_resized_crop.scale = (0.5, 1.0)
_C.transforms.img.random_resized_crop.ratio = (3/4, 4/3)

# resize
_C.transforms.img.resize =  CN()
_C.transforms.img.resize.enable = 1

# random_crop
_C.transforms.img.random_crop = CN()
_C.transforms.img.random_crop.enable = 1
_C.transforms.img.random_crop.padding = 0

# center_crop
_C.transforms.img.center_crop = CN()
_C.transforms.img.center_crop.enable = 0

### without modifying the image size
_C.transforms.img.aug_imagenet = False
_C.transforms.img.aug_cifar = False

# color_jitter
_C.transforms.img.color_jitter = CN()
_C.transforms.img.color_jitter.enable = 0
_C.transforms.img.color_jitter.brightness = 0.
_C.transforms.img.color_jitter.contrast = 0.
_C.transforms.img.color_jitter.saturation = 0.
_C.transforms.img.color_jitter.hue = 0.

# horizontal_flip
_C.transforms.img.random_horizontal_flip = CN()
_C.transforms.img.random_horizontal_flip.enable = 1
_C.transforms.img.random_horizontal_flip.p = 0.5

# vertical_flip
_C.transforms.img.random_vertical_flip = CN()
_C.transforms.img.random_vertical_flip.enable = 1
_C.transforms.img.random_vertical_flip.p = 0.5

# random_rotation
_C.transforms.img.random_rotation = CN()
_C.transforms.img.random_rotation.enable = 1
_C.transforms.img.random_rotation.degrees = 10



_C.label_transforms = CN() # label transforms
_C.label_transforms.name = 'default'


# ---------------------------------------------------------------------------- #
# dataloader
# ---------------------------------------------------------------------------- #
_C.dataloader = CN()
_C.dataloader.num_workers = 4
_C.dataloader.sample_train = "default"
_C.dataloader.sample_test = "default"


# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #
_C.model = CN()
_C.model.meta_arch = 'Resnet50'
_C.model.WEIGHTS = ""
_C.model.classes = 10
_C.model.pretrained = True
_C.model.finetune = False
_C.model.features = ['f4', ]
_C.model.features_fusion = 'sum'


# ---------------------------------------------------------------------------- #
# optimizer
# ---------------------------------------------------------------------------- #
_C.optim = CN()
_C.optim.name = 'adam'
_C.optim.momentum = 0.9
_C.optim.base_lr = 0.001
_C.optim.weight_decay = 0.0005

# scheduler
_C.optim.scheduler = CN()
_C.optim.scheduler.name = 'MultiStepLR'
_C.optim.scheduler.gamma = 0.1 # decay factor

# for CosineAnnealingLR
_C.optim.scheduler.t_max = 10 

# for ReduceLROnPlateau
_C.optim.scheduler.mode = 'min' # min for loss, max for acc
_C.optim.scheduler.patience = 10
_C.optim.scheduler.verbose = True # print log once update lr

# for StepLR
_C.optim.scheduler.step_size = 10

# for MultiStepLR
_C.optim.scheduler.milestones = [10, 25, 35, 50]

# ---------------------------------------------------------------------------- #
# loss
# ---------------------------------------------------------------------------- #
_C.loss = CN()
_C.loss.name = 'CrossEntropy'
_C.loss.class_weight = []

# ---------------------------------------------------------------------------- #
# hooks
# ---------------------------------------------------------------------------- #
_C.hooks = CN()

## EarlyStopping
_C.hooks.early-stopping = CN()
_C.hooks.early-stopping.type = 2 # 0: True 1: False 2: custom
_C.hooks.early-stopping.monitor = 'val_loss'
_C.hooks.early-stopping.min_delta = 0.
_C.hooks.early-stopping.patience = 10
_C.hooks.early-stopping.mode = 'min'
_C.hooks.early-stopping.verbose = 1

# ModelCheckpoint
_C.hooks.model_checkpoint = CN()
_C.hooks.model_checkpoint.type = 0 # 0: True 1: False 2: custom
_C.hooks.model_checkpoint.filepath = '' # the empty file path is recommended
_C.hooks.model_checkpoint.monitor = 'val_loss'
_C.hooks.model_checkpoint.mode = 'min'
_C.hooks.model_checkpoint.verbose = 1


# ---------------------------------------------------------------------------- #
# Module template 
# ---------------------------------------------------------------------------- #

_C.module_template = CN()
_C.module_template.name = 'LightningTemplateModel'

# ---------------------------------------------------------------------------- #
# Trainer 
# ---------------------------------------------------------------------------- #

_C.trainer = CN()
_C.trainer.ACCUMULATE_GRAD_BATCHES = 1
_C.trainer.min_epochs = 30
_C.trainer.MAX_EPOCHS = 1000
_C.trainer.grad_clip_val = 0.5 # clip gradient of which norm is larger than 0.5
_C.trainer.show_progress_bar = True # show progree bar
_C.trainer.row_log_interval = 100 # Every k batches lightning will make an entry in the metrics log
_C.trainer.log_save_interval = 100 # Every k batches, lightning will write the new logs to disk, ie: save a .csv log file every 100 batches
_C.trainer.default_save_path = './output'
_C.trainer.log_gpu_memory = "" # 'min_max': log only the min/max utilization
_C.trainer.fast_dev_run = False # everything only with 1 training and 1 validation batch.

_C.trainer.logger = CN()
_C.trainer.logger.setting = 0 # 0: True  1: False  2: custom
_C.trainer.logger.type = 'mlflow'
_C.trainer.logger.mlflow = CN()
_C.trainer.logger.mlflow.experiment_name = 'torchline_logs'
_C.trainer.logger.mlflow.tracking_uri = _C.trainer.default_save_path
_C.trainer.logger.test_tube = CN()
_C.trainer.logger.test_tube.name = 'torchline_logs'
_C.trainer.logger.test_tube.save_dir = _C.trainer.default_save_path
_C.trainer.logger.test_tube.debug = False
_C.trainer.logger.test_tube.version = -1 #  # if <0, then use default version. Otherwise, it will restore the version.


# ---------------------------------------------------------------------------- #
# log
# ---------------------------------------------------------------------------- #

_C.log = CN()
_C.log.name = 'log.txt'

# ---------------------------------------------------------------------------- #
# Misc 
# ---------------------------------------------------------------------------- #

_C.SEED = 666
_C.DEFAULT_CUDNN_BENCHMARK = True

_C.TOPK = [1, 3] # save the top k results., e.g. acc@1 and acc@3

_C.predict_only = CN()
_C.predict_only.type = 'ckpt'
_C.predict_only.to_pred_file_path = '' # specify the path of images

_C.predict_only.load_ckpt = CN() # load from checkpoint
_C.predict_only.load_ckpt.checkpoint_path = '' # load_from_checkpoint

_C.predict_only.load_metric = CN()
_C.predict_only.load_metric.weights_path = '' # load_from_metrics
_C.predict_only.load_metric.tags_csv = ''
_C.predict_only.load_metric.on_gpu = True
_C.predict_only.load_metric.map_location = 'cuda:0'

