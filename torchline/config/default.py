import random
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


# ---------------------------------------------------------------------------- #
# albumentations transforms (abtfs)
# ---------------------------------------------------------------------------- #

_C.abtfs = CN()
_C.abtfs.random_grid_shuffle = CN()
_C.abtfs.random_grid_shuffle.enable = 0
_C.abtfs.random_grid_shuffle.grid = 2

_C.abtfs.channel_shuffle = CN()
_C.abtfs.channel_shuffle.enable = 0

_C.abtfs.channel_dropout = CN()
_C.abtfs.channel_dropout.enable = 0
_C.abtfs.channel_dropout.drop_range = (1, 1)
_C.abtfs.channel_dropout.fill_value = 127

_C.abtfs.noise = CN()
_C.abtfs.noise.enable = 1

_C.abtfs.blur = CN()
_C.abtfs.blur.enable = 0

_C.abtfs.rotate = CN()
_C.abtfs.rotate.enable = 1

_C.abtfs.bright = CN()
_C.abtfs.bright.enable = 1

_C.abtfs.distortion = CN()
_C.abtfs.distortion.enable = 0

_C.abtfs.hue = CN()
_C.abtfs.hue.enable = 0

_C.abtfs.cutout = CN()
_C.abtfs.cutout.enable = 1
_C.abtfs.cutout.num_holes = 10
_C.abtfs.cutout.size = 20
_C.abtfs.cutout.fill_value = 127

# ---------------------------------------------------------------------------- #
# torchvision transforms
# ---------------------------------------------------------------------------- #

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
_C.model.name = 'Resnet50'
_C.model.classes = 10
_C.model.pretrained = True
_C.model.finetune = False


# ---------------------------------------------------------------------------- #
# optimizer
# ---------------------------------------------------------------------------- #
_C.optim = CN()
_C.optim.name = 'adam'
_C.optim.momentum = 0.9
_C.optim.base_lr = 0.001
# _C.optim.lr = _C.optim.base_lr # will changed in v0.3.0.0
_C.optim.weight_decay = 0.0005

# scheduler
_C.optim.scheduler = CN()
_C.optim.scheduler.name = 'MultiStepLR'
_C.optim.scheduler.gamma = 0.1 # decay factor

# for CosineAnnealingLR
_C.optim.scheduler.t_max = 10 

# for CosineAnnealingLR
_C.optim.scheduler.t_0 = 5
_C.optim.scheduler.t_mul = 20

# for ReduceLROnPlateau
_C.optim.scheduler.mode = 'min' # min for loss, max for acc
_C.optim.scheduler.patience = 10
_C.optim.scheduler.verbose = True # print log once update lr

# for StepLR
_C.optim.scheduler.step_size = 10

# for MultiStepLR
_C.optim.scheduler.milestones = [10, 25, 35, 50]

# _C.optimizer = _C.optim # enhance compatibility. will changed in v0.3.0.0
# ---------------------------------------------------------------------------- #
# loss
# ---------------------------------------------------------------------------- #
_C.loss = CN()
_C.loss.name = 'CrossEntropy'
_C.loss.class_weight = []
_C.loss.label_smoothing = 0.1 # CrossEntropyLabelSmooth

_C.loss.focal_loss = CN()
_C.loss.focal_loss.alpha = [] # FocalLoss
_C.loss.focal_loss.gamma = 2
_C.loss.focal_loss.size_average = True
# ---------------------------------------------------------------------------- #
# hooks
# ---------------------------------------------------------------------------- #
_C.hooks = CN()

## EarlyStopping
_C.hooks.early_stopping = CN()
_C.hooks.early_stopping.setting = 2 # 0: True 1: False 2: custom
_C.hooks.early_stopping.monitor = 'valid_loss'
_C.hooks.early_stopping.min_delta = 0.
_C.hooks.early_stopping.patience = 10
_C.hooks.early_stopping.mode = 'min'
_C.hooks.early_stopping.verbose = 1

# ModelCheckpoint
_C.hooks.model_checkpoint = CN()
_C.hooks.model_checkpoint.setting = 0 # 0: True 1: False 2: custom
_C.hooks.model_checkpoint.filepath = '' # the empty file path is recommended
_C.hooks.model_checkpoint.monitor = 'valid_loss'
_C.hooks.model_checkpoint.mode = 'min'
_C.hooks.model_checkpoint.verbose = 1


# ---------------------------------------------------------------------------- #
# Module template 
# ---------------------------------------------------------------------------- #

_C.module = CN()
_C.module.name = 'DefaultModule'

# ---------------------------------------------------------------------------- #
# Trainer 
# ---------------------------------------------------------------------------- #

_C.trainer = CN()
_C.trainer.name = 'DefaultTrainer'
_C.trainer.default_save_path = './output'
_C.trainer.gradient_clip_val = 0
_C.trainer.process_position = 0
_C.trainer.num_nodes = 1
_C.trainer.gpus = [] # list
_C.trainer.log_gpu_memory = ""
_C.trainer.show_progress_bar = False
_C.trainer.overfit_pct = 0.0 # if 0<overfit_pct<1, (e.g. overfit_pct = 0.1) then train, val, test only 10% data.
_C.trainer.track_grad_norm = -1 # -1 no tracking. Otherwise tracks that norm. if equals to 2, then 2-norm will be traced
_C.trainer.check_val_every_n_epoch = 1
_C.trainer.fast_dev_run = False # everything only with 1 training and 1 validation batch.
_C.trainer.accumulate_grad_batches = 1
_C.trainer.max_epochs = 100
_C.trainer.min_epochs = 1
_C.trainer.train_percent_check = 1.0
_C.trainer.val_percent_check = 1.0
_C.trainer.test_percent_check = 1.0
_C.trainer.val_check_interval = 1.0
_C.trainer.log_save_interval = 100 # Writes logs to disk this often
_C.trainer.row_log_interval = 10 # How often to add logging rows (does not write to disk)
_C.trainer.distributed_backend = 'dp' # 'dp', 'ddp', 'ddp2'.
_C.trainer.use_amp = False
_C.trainer.print_nan_grads = True # Prints gradients with nan values
_C.trainer.weights_summary = '' # 'full', 'top', None.
_C.trainer.weights_save_path = ''
_C.trainer.amp_level = 'O1'
_C.trainer.num_sanity_val_steps = 5
_C.trainer.truncated_bptt_steps = ''
_C.trainer.resume_from_checkpoint = ''
# _C.trainer.profiler = ''


_C.trainer.logger = CN()
_C.trainer.logger.type = 'test_tube'
_C.trainer.logger.setting = 0 # 0: True  1: False  2: custom
_C.trainer.logger.mlflow = CN()
_C.trainer.logger.mlflow.experiment_name = 'torchline_logs'
_C.trainer.logger.mlflow.tracking_uri = _C.trainer.default_save_path
_C.trainer.logger.test_tube = CN()
_C.trainer.logger.test_tube.name = 'torchline_logs'
_C.trainer.logger.test_tube.save_dir = _C.trainer.default_save_path
_C.trainer.logger.test_tube.version = -1 #  # if <0, then use default version. Otherwise, it will restore the version.


# ---------------------------------------------------------------------------- #
# log
# ---------------------------------------------------------------------------- #

_C.log = CN()
_C.log.path = ''
_C.log.name = 'log.txt'

# ---------------------------------------------------------------------------- #
# Misc 
# ---------------------------------------------------------------------------- #

_C.SEED = random.randint(0, 10000)
_C.DEFAULT_CUDNN_BENCHMARK = True

_C.topk = [1, 3] # save the top k results., e.g. acc@1 and acc@3

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

