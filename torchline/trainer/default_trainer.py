
import os
import shutil

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from torchline.utils import Logger

from .build import TRAINER_REGISTRY

__all__ = [
    'DefaultTrainer'
]


@TRAINER_REGISTRY.register()
class DefaultTrainer(Trainer):
    def __init__(self, cfg, hparams):
        self.cfg = cfg
        self.hparams = hparams

        self.logger = self._logger()
        self.logger_print = Logger(__name__, cfg).getlogger()
        self.early_stop_callback = self._early_stop_callback()
        self.checkpoint_callback = self._checkpoint_callback()

        # you can update trainer_params to change different parameters
        self.trainer_params = dict(self.cfg.trainer)
        self.trainer_params.update({
            'logger': self.logger,
            'early_stop_callback': self.early_stop_callback,
            'checkpoint_callback': self.checkpoint_callback,
        })
        self.trainer_params.pop('name')
        for key in self.trainer_params:
            self.trainer_params[key] = self.parse_cfg_param(self.trainer_params[key])
            
        super(DefaultTrainer, self).__init__(**self.trainer_params)

    def parse_cfg_param(self, cfg_item):
        return cfg_item if cfg_item not in ['', []] else None

    def _logger(self):

        # logger
        logger_cfg = self.cfg.trainer.logger
        assert logger_cfg.setting in [0,1,2], "You can only set three logger levels [0,1,2], but you set {}".format(logger_cfg.setting)
        if logger_cfg.type == 'mlflow':
            raise NotImplementedError
            # params = {key: logger_cfg.mlflow[key] for key in logger_cfg.mlflow}
            # custom_logger = MLFlowLogger(**params)
        elif logger_cfg.type == 'test_tube':
            params = {key: logger_cfg.test_tube[key] for key in logger_cfg.test_tube} # key: save_dir, name, version

            # save_dir: logger root path: 
            if self.cfg.trainer.default_save_path:
                save_dir = self.cfg.trainer.default_save_path
            else:
                save_dir = logger_cfg.test_tube.save_dir
            
            # version
            if logger_cfg.setting==1:
                return False
            else:
                version = self.cfg.log.path[-1]

            default_logger = TestTubeLogger(save_dir, name='torchline_logs', version=version)
            params.update({'version': version, 'name':logger_cfg.test_tube.name, 'save_dir': save_dir})
            custom_logger = TestTubeLogger(**params)
        else:
            print(f"{logger_cfg.type} not supported")
            raise NotImplementedError
        
        loggers = {
            0: default_logger,
            1: False,
            2: custom_logger
        } # 0: True (default)  1: False  2: custom
        logger = loggers[logger_cfg.setting]

        return logger

    def _early_stop_callback(self):
        # early_stop_callback hooks
        hooks = self.cfg.hooks
        params = {key: hooks.early_stopping[key] for key in hooks.early_stopping if key != 'setting'}
        early_stop_callbacks = {
            0: True,  # default setting
            1: False, # do not use early stopping
            2: EarlyStopping(**params) # use custom setting
        }
        assert hooks.early_stopping.setting in early_stop_callbacks, 'The setting of early stopping can only be in [0,1,2]'
        early_stop_callback = early_stop_callbacks[hooks.early_stopping.setting]
        return early_stop_callback

    def _checkpoint_callback(self):
        # checkpoint_callback hooks
        hooks = self.cfg.hooks
        assert hooks.model_checkpoint.setting in [0,1,2], "You can only set three ckpt levels [0,1,2], but you set {}".format(hooks.model_checkpoint.setting)

        params = {key: hooks.model_checkpoint[key] for key in hooks.model_checkpoint if key != 'setting'}
        if hooks.model_checkpoint.setting==2:
            if hooks.model_checkpoint.filepath.strip()=='':
                filepath = os.path.join(self.cfg.log.path,'checkpoints')
                params.update({'filepath': filepath})
            else:
                self.logger_print.warn("The specified checkpoint path is not recommended!")
        checkpoint_callbacks = {
            0: True,
            1: False,
            2: ModelCheckpoint(**params)
        }
        checkpoint_callback = checkpoint_callbacks[hooks.model_checkpoint.setting]
        return checkpoint_callback
