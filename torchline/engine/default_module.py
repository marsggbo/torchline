"""
Example template for defining a system
"""
import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision

from torchline.data import build_data, build_sampler
from torchline.losses import build_loss_fn
from torchline.models import build_model
from torchline.utils import topk_acc, AverageMeterGroup
from .build import MODULE_REGISTRY
from .utils import generate_optimizer, generate_scheduler

__all__ = [
    'DefaultModule'
]

@MODULE_REGISTRY.register()
class DefaultModule(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, cfg):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param cfg:
        """
        # init superclass
        super(DefaultModule, self).__init__()
        self.cfg = cfg
        self.hparams = self.cfg.hparams
        self.batch_size = self.cfg.dataset.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        h, w = self.cfg.input.size
        self.example_input_array = torch.rand(1, 3, h, w)

        # build model
        self.model = self.build_model(cfg)
        self.train_meters = AverageMeterGroup()
        self.valid_meters = AverageMeterGroup()

    # ---------------------
    # model SETUP
    # ---------------------
    def build_model(self, cfg):
        """
        Layout model
        :return:
        """
        return build_model(cfg)

    def build_loss_fn(self, cfg):
        """
        Layout loss_fn
        :return:
        """
        return build_loss_fn(cfg)

    def build_data(self, cfg, is_train):
        """
        Layout training dataset
        :return:
        """
        cfg.defrost()
        cfg.dataset.is_train = is_train
        cfg.freeze()
        return build_data(cfg)

    def build_sampler(self, cfg, is_train):
        """
        Layout training dataset
        :return:
        """
        cfg.defrost()
        cfg.dataset.is_train = is_train
        cfg.freeze()
        return build_sampler(cfg)

    # ---------------------
    # Hooks
    # ---------------------

    def on_train_start(self):
        ckpt_path = self.trainer.resume_from_checkpoint
        print(ckpt_path)
        if ckpt_path:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path)
                best = ckpt['checkpoint_callback_best']
                self.trainer.logger_print.info(f"The best result of the ckpt is {best}")
            else:
                print(f'{ckpt_path} not exists')
                raise NotImplementedError

    def on_epoch_start(self):
        if not self.cfg.trainer.show_progress_bar:
            # print current lr
            if isinstance(self.trainer.optimizers, list):
                if len(self.trainer.optimizers) == 1:
                    optimizer = self.trainer.optimizers[0]
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"lr={lr:.4e}")
                else:
                    for index, optimizer in enumerate(self.trainer.optimizers):
                        lr = optimizer.param_groups[0]["lr"]
                        name = str(optimizer).split('(')[0].strip()
                        self.trainer.logger_print.info(f"lr of {name}_{index} is {lr:.4e} ")
            else:
                lr = self.trainer.optimizers.param_groups[0]["lr"]
                print(f"lr={lr:.4e}")

    def on_epoch_end(self):
        if not self.cfg.trainer.show_progress_bar:
            self.trainer.logger_print.info(f'Final Train: {self.train_meters}')
            self.trainer.logger_print.info(f'FInal Valid: {self.valid_meters}')
            self.trainer.logger_print.info("===========================\n")
            self.train_meters = AverageMeterGroup()
            self.valid_meters = AverageMeterGroup()

    # ---------------------
    # TRAINING
    # ---------------------
    
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        
        return middle features
        features = self.model.features(x)
        logits = self.model.logits(features)
        return logits
        """
        return self.model(x)

    def loss(self, predictions, gt_labels):
        loss_fn = self.build_loss_fn(self.cfg)
        return loss_fn(predictions, gt_labels)

    def print_log(self, batch_idx, is_train, inputs, meters, save_examples=False):
        if is_train:
            _type = 'Train'
            all_step = self.trainer.num_training_batches
        else:
            _type = 'Valid'
            all_step = self.trainer.total_batches - self.trainer.num_training_batches

        flag = batch_idx % self.cfg.trainer.log_save_interval == 0
        if not self.trainer.show_progress_bar and flag:
            crt_epoch, crt_step = self.trainer.current_epoch, batch_idx
            all_epoch = self.trainer.max_epochs
            log_info = f"{_type} Epoch {crt_epoch}/{all_epoch} step {crt_step}/{all_step} {meters}" 
            self.trainer.logger_print.info(log_info)

        if self.current_epoch==0 and batch_idx==0 and save_examples:
            if not os.path.exists('train_valid_samples'):
                os.makedirs('train_valid_samples')
            for i, img in enumerate(inputs[:5]):
                torchvision.transforms.ToPILImage()(img.cpu()).save(f'./train_valid_samples/{_type}_img{i}.jpg')

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """

        # forward pass
        inputs, gt_labels = batch
        predictions = self.forward(inputs)

        # calculate loss
        loss_val = self.loss(predictions, gt_labels)

        # acc
        acc_results = topk_acc(predictions, gt_labels, self.cfg.topk)
        tqdm_dict = {}

        if self.on_gpu:
            acc_results = [torch.tensor(x).to(loss_val.device.index) for x in acc_results]

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            acc_results = [x.unsqueeze(0) for x in acc_results]

        tqdm_dict['train_loss'] = loss_val
        for i, k in enumerate(self.cfg.topk):
            tqdm_dict[f'train_acc_{k}'] = acc_results[i]

        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self.train_meters.update({key: val.item() for key, val in tqdm_dict.items()})
        self.print_log(batch_idx, True, inputs, self.train_meters)

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        inputs, gt_labels = batch
        predictions = self.forward(inputs)

        loss_val = self.loss(predictions, gt_labels)

        # acc
        val_acc_1, val_acc_k = topk_acc(predictions, gt_labels, self.cfg.topk)

        if self.on_gpu:
            val_acc_1 = val_acc_1.cuda(loss_val.device.index)
            val_acc_k = val_acc_k.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc_1 = val_acc_1.unsqueeze(0)
            val_acc_k = val_acc_k.unsqueeze(0)
        
        output = OrderedDict({
            'valid_loss': loss_val,
            'valid_acc_1': val_acc_1,
            f'valid_acc_{self.cfg.topk[-1]}': val_acc_k,
        })
        tqdm_dict = {k: v for k, v in dict(output).items()}
        self.valid_meters.update({key: val.item() for key, val in tqdm_dict.items()})
        self.print_log(batch_idx, False, inputs, self.valid_meters)

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        tqdm_dict = {key: val.avg for key, val in self.valid_meters.meters.items()}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'valid_loss': self.valid_meters.meters['valid_loss'].avg}
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_end(self, outputs):
        return self.validation_end(outputs)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    @classmethod
    def parse_cfg_for_scheduler(cls, cfg, scheduler_name):
        if scheduler_name.lower() == 'CosineAnnealingLR'.lower():
            params = {'T_max': cfg.optim.scheduler.t_max}
        elif scheduler_name.lower() == 'CosineAnnealingWarmRestarts'.lower():
            params = {'T_0': cfg.optim.scheduler.t_0, 'T_mult': cfg.optim.scheduler.t_mul}
        elif scheduler_name.lower() == 'StepLR'.lower():
            params = {'step_size': cfg.optim.scheduler.step_size, 'gamma': cfg.optim.scheduler.gamma}
        elif scheduler_name.lower() == 'MultiStepLR'.lower():
            params = {'milestones': cfg.optim.scheduler.milestones, 'gamma': cfg.optim.scheduler.gamma}
        elif scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
            params = {'mode': cfg.optim.scheduler.mode, 'patience': cfg.optim.scheduler.patience, 
                      'verbose': cfg.optim.scheduler.verbose, 'factor': cfg.optim.scheduler.gamma}
        else:
            print(f"{scheduler_name} not implemented")
            raise NotImplementedError
        return params

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optim_name = self.cfg.optim.name
        momentum = self.cfg.optim.momentum
        weight_decay = self.cfg.optim.weight_decay
        lr = self.cfg.optim.base_lr
        optimizer = generate_optimizer(self.model, optim_name, lr, momentum, weight_decay)
        scheduler_params = self.parse_cfg_for_scheduler(self.cfg, self.cfg.optim.scheduler.name)
        scheduler = generate_scheduler(optimizer, self.cfg.optim.scheduler.name, **scheduler_params)
        return [optimizer], [scheduler]

    def __dataloader(self, is_train):
        # init data generators
        dataset = self.build_data(self.cfg, is_train)

        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = self.build_sampler(self.cfg, is_train)
        batch_size = self.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.cfg.dataloader.num_workers
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        logging.info('training data loader called')
        return self.__dataloader(is_train=True)

    @pl.data_loader
    def val_dataloader(self):
        logging.info('val data loader called')
        return self.__dataloader(is_train=False)

    @pl.data_loader
    def test_dataloader(self):
        logging.info('test data loader called')
        return self.__dataloader(is_train=False)
