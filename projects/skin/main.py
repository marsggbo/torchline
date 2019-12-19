"""
Runs a model on a single node across N-gpus.
"""
import argparse
import glob
import os
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import MLFlowLogger, TestTubeLogger

from config import add_skin_config
from data import *
from losses import FocalLoss
from models import *
from torchline.config import get_cfg
from torchline.engine import build_module_template
from torchline.utils import Logger, get_imgs_to_predict

logger_print = Logger(__name__).getlog()

def parse_cfg_param(cfg_item):
    return cfg_item if cfg_item else None

def setup(args):
    """
    Create configs and perform basic setups.
    """
    assert not (args.test_only and args.predict_only), "You can't set both 'test_only' and 'predict_only' True"
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    add_skin_config(cfg)
    cfg.merge_from_list(args.opts)
    cfg.update({'hparams': args})
    cfg.freeze()
    
    if hasattr(args, "config_file"):
        logger_print.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, open(args.config_file, "r").read()
            )
        )
    
    SEED = cfg.SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not (args.test_only or args.predict_only):
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg.DEFAULT_CUDNN_BENCHMARK

    
    logger_print.info("Running with full config:\n{}".format(cfg))
    return cfg

class MyTrainer(Trainer):
    def __init__(self, cfg, hparams):
        self.cfg = cfg
        self.hparams = hparams
        self.resume_from_checkpoint = None 
        if hparams.resume:
            self.resume_from_checkpoint = hparams.resume

        self.logger = self._logger()
        self.early_stop_callback = self._early_stop_callback()
        self.checkpoint_callback = self._checkpoint_callback()

        # you can update trainer_params to change different parameters
        self.trainer_params = {
            'gpus': hparams.gpus,
            'use_amp': hparams.use_16bit,
            'distributed_backend': hparams.distributed_backend,

            'min_epochs' : cfg.TRAINER.MIN_EPOCHS,
            'max_epochs' : cfg.TRAINER.MAX_EPOCHS,
            'gradient_clip_val' : cfg.TRAINER.GRAD_CLIP_VAL,
            'show_progress_bar' : cfg.TRAINER.SHOW_PROGRESS_BAR,
            'row_log_interval' : cfg.TRAINER.ROW_LOG_INTERVAL,
            'log_save_interval' : cfg.TRAINER.LOG_SAVE_INTERVAL,
            'log_gpu_memory' : parse_cfg_param(cfg.TRAINER.LOG_GPU_MEMORY),
            'default_save_path' : cfg.TRAINER.DEFAULT_SAVE_PATH,
            'fast_dev_run' : cfg.TRAINER.FAST_DEV_RUN,

            'logger': self.logger,
            'early_stop_callback': self.early_stop_callback,
            'checkpoint_callback': self.checkpoint_callback,
            'resume_from_checkpoint': self.resume_from_checkpoint,

            'weights_summary': None,
        }

        super(MyTrainer, self).__init__(**self.trainer_params)
    
    def _logger(self):
        def _version_logger(save_dir, logger_name):
            path = os.path.join(save_dir, logger_name)
            if (not os.path.exists(path)) or len(os.listdir(path))==0:
                version = 0
            else:
                versions = [int(v.split('_')[-1]) for v in os.listdir(path)]
                version = max(versions)+1
            return version

        # logger
        LOGGER = self.cfg.TRAINER.LOGGER
        assert LOGGER.SETTING in [0,1,2], "You can only set three logger levels [0,1,2], but you set {}".format(LOGGER.SETTING)
        if LOGGER.type == 'mlflow':
            params = {key: LOGGER.MLFLOW[key] for key in LOGGER.MLFLOW}
            custom_logger = MLFlowLogger(**params)
        elif LOGGER.type == 'test_tube':
            params = {key: LOGGER.TEST_TUBE[key] for key in LOGGER.TEST_TUBE} # key: save_dir, name, version

            # save_dir: logger root path: 
            if self.cfg.TRAINER.DEFAULT_SAVE_PATH:
                save_dir = self.cfg.TRAINER.DEFAULT_SAVE_PATH
            else:
                save_dir = LOGGER.TEST_TUBE.save_dir
            
            # version
            if LOGGER.SETTING==0:
                version = _version_logger(save_dir, 'torchline_logs')
            elif LOGGER.SETTING==2:
                if LOGGER.TEST_TUBE.version<0:
                    version = _version_logger(save_dir, LOGGER.TEST_TUBE.name)
                else:
                    version = int(LOGGER.TEST_TUBE.version)
            else:
                return False

            default_logger = TestTubeLogger(save_dir, name='torchline_logs', version=version)
            params.update({'version': version, 'save_dir': save_dir})
            custom_logger = TestTubeLogger(**params)
        else:
            print(f"{LOGGER.type} not supported")
            raise NotImplementedError
        
        loggers = {
            0: default_logger,
            1: False,
            2: custom_logger
        } # 0: True (default)  1: False  2: custom
        logger = loggers[LOGGER.SETTING]

        # copy config file to the logger directory
        if LOGGER.SETTING!=1:
            src_cfg_file = self.hparams.config_file # source config file
            cfg_file_name = os.path.basename(src_cfg_file) # config file name
            dst_cfg_file = os.path.join(self.cfg.TRAINER.DEFAULT_SAVE_PATH, logger.name, f"version_{logger.version}")
            if not os.path.exists(dst_cfg_file):
                os.makedirs(dst_cfg_file)
            dst_cfg_file = os.path.join(dst_cfg_file, cfg_file_name)
            shutil.copy(src_cfg_file, dst_cfg_file)
        return logger

    def _early_stop_callback(self):
        # early_stop_callback hooks
        HOOKS = self.cfg.HOOKS
        params = {key: HOOKS.EARLY_STOPPING[key] for key in HOOKS.EARLY_STOPPING if key != 'type'}
        early_stop_callbacks = {
            0: True,  # default setting
            1: False, # do not use early stopping
            2: EarlyStopping(**params) # use custom setting
        }
        assert HOOKS.EARLY_STOPPING.type in early_stop_callbacks, 'The type of early stopping can only be in [0,1,2]'
        early_stop_callback = early_stop_callbacks[HOOKS.EARLY_STOPPING.type]
        return early_stop_callback

    def _checkpoint_callback(self):
        # checkpoint_callback hooks
        HOOKS = self.cfg.HOOKS
        assert HOOKS.MODEL_CHECKPOINT.type in [0,1,2], "You can only set three ckpt levels [0,1,2], but you set {}".format(HOOKS.MODEL_CHECKPOINT.type)

        logger = self.logger
        params = {key: HOOKS.MODEL_CHECKPOINT[key] for key in HOOKS.MODEL_CHECKPOINT if key != 'type'}
        if HOOKS.MODEL_CHECKPOINT.type==2:
            if HOOKS.MODEL_CHECKPOINT.filepath.strip()=='':
                filepath = os.path.join(self.cfg.TRAINER.DEFAULT_SAVE_PATH, logger.name,
                                f'version_{logger.version}','checkpoints')
                params.update({'filepath': filepath})
            else:
                logger_print.warn("The specified checkpoint path is not recommended!")
        checkpoint_callbacks = {
            0: True,
            1: False,
            2: ModelCheckpoint(**params)
        }
        checkpoint_callback = checkpoint_callbacks[HOOKS.MODEL_CHECKPOINT.type]
        return checkpoint_callback

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    cfg = setup(hparams)

    # only predict on some samples
    if hasattr(hparams, "predict_only") and hparams.predict_only:
        PREDICT_ONLY = cfg.PREDICT_ONLY
        if PREDICT_ONLY.type == 'ckpt':
            load_params = {key: PREDICT_ONLY.LOAD_CKPT[key] for key in PREDICT_ONLY.LOAD_CKPT}
            model = build_module_template(cfg)
            ckpt_path = load_params['checkpoint_path']
            model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        elif PREDICT_ONLY.type == 'metrics':
            load_params = {key: PREDICT_ONLY.LOAD_METRIC[key] for key in PREDICT_ONLY.LOAD_METRIC}
            model = build_module_template(cfg).load_from_metrics(**load_params)
        else:
            print(f'{cfg.PREDICT_ONLY.type} not supported')
            raise NotImplementedError

        model.eval()
        model.freeze() 
        images = get_imgs_to_predict(cfg.PREDICT_ONLY.to_pred_file_path, cfg)
        if torch.cuda.is_available():
            images['img_data'] = images['img_data'].cuda()
            model = model.cuda()
        if cfg.MODEL.CLASSES == 10:
            classes = np.loadtxt('skin10_class_names.txt', dtype=str)
        else:
            classes = np.loadtxt('skin100_class_names.txt', dtype=str)
        predictions = model(images['img_data'])
        class_indices = torch.argmax(predictions, dim=1)
        for i, file in enumerate(images['img_file']):
            index = class_indices[i]
            print(f"{file} is {classes[index]}")
        return predictions.cpu()
    elif hasattr(hparams, "test_only") and hparams.test_only:
        model = build_module_template(cfg)
        trainer = MyTrainer(cfg, hparams)
        trainer.test(model)
    else:
        model = build_module_template(cfg)
        trainer = MyTrainer(cfg, hparams)
        trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        "--config_file", 
        default="", 
        metavar="FILE", 
        help="path to config file"
    )
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='resume_from_checkpoint'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )
    parent_parser.add_argument(
        '--test_only',
        action='store_true',
        help='if true, return trainer.test(model). Validates only the test set'
    )
    parent_parser.add_argument(
        '--predict_only',
        action='store_true',
        help='if true run model(samples). Predict on the given samples.'
    )
    parent_parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # each LightningModule defines arguments relevant to it
    hparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
