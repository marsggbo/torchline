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

    # ---------------------
    # model SETUP
    # ---------------------
    @classmethod
    def build_model(cls, cfg):
        """
        Layout model
        :return:
        """
        return build_model(cfg)

    @classmethod
    def build_loss_fn(cls, cfg):
        """
        Layout loss_fn
        :return:
        """
        return build_loss_fn(cfg)

    @classmethod
    def build_data(cls, cfg, is_train):
        """
        Layout training dataset
        :return:
        """
        cfg.defrost()
        cfg.dataset.is_train = is_train
        cfg.freeze()
        return build_data(cfg)

    @classmethod
    def build_sampler(cls, cfg, is_train):
        """
        Layout training dataset
        :return:
        """
        cfg.defrost()
        cfg.dataset.is_train = is_train
        cfg.freeze()
        return build_sampler(cfg)

    # ---------------------
    # TRAINING
    # ---------------------
    
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """

        logits = self.model(x)
        return logits

    def loss(self, predictions, gt_labels):
        loss_fn = self.build_loss_fn(self.cfg)
        return loss_fn(predictions, gt_labels)

    def on_epoch_end(self):
        self.trainer.logger_print.info(f'Final Train: {self.train_meters}')
        self.trainer.logger_print.info(f'FInal Valid: {self.valid_meters}')
        self.trainer.logger_print.info("===========================\n")

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
        if batch_idx == 0:
            self.train_meters = AverageMeterGroup() # reset meters at a new epoch

        # forward pass
        inputs, gt_labels = batch
        predictions = self.forward(inputs)

        # calculate loss
        loss_val = self.loss(predictions, gt_labels)

        # acc
        acc_results = topk_acc(predictions, gt_labels, self.cfg.topk)
        tqdm_dict = {}

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict.update({'train_loss': loss_val.item()})
        for i, k in enumerate(self.cfg.topk):
            tqdm_dict[f'train_acc_{k}'] = acc_results[i].item()

        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self.train_meters.update(tqdm_dict)
        self.print_log(batch_idx, True, inputs, self.train_meters)

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        if batch_idx == 0:
            self.valid_meters = AverageMeterGroup() # reset meters at a new epoch
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
        tqdm_dict = {k: v.item() for k, v in dict(output).items()}
        self.valid_meters.update(tqdm_dict)
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

        val_loss_mean = 0
        val_acc_1_mean = 0
        val_acc_k_mean = 0
        for output in outputs:
            val_loss = output['valid_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc_1 = output['valid_acc_1']
            val_acc_k = output[f'valid_acc_{self.cfg.topk[-1]}']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc_1 = torch.mean(val_acc_1)
                val_acc_k = torch.mean(val_acc_k)

            val_acc_1_mean += val_acc_1
            val_acc_k_mean += val_acc_k

        val_loss_mean /= len(outputs)
        val_acc_1_mean /= len(outputs)
        val_acc_k_mean /= len(outputs)
        tqdm_dict = {'valid_loss': val_loss_mean, 'valid_acc_1': val_acc_1_mean, 
                                                f'valid_acc_{self.cfg.topk[-1]}': val_acc_k_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'valid_loss': val_loss_mean}
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_end(self, outputs):
        return self.validation_end(outputs)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def generate_optimizer(self):
        '''
        return torch.optim.Optimizer
        '''
        optim_name = self.cfg.optim.name
        momentum = self.cfg.optim.momentum
        weight_decay = self.cfg.optim.weight_decay
        lr = self.cfg.optim.base_lr
        if optim_name.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optim_name.lower() == 'adadelta':
            return torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim_name.lower() == 'adam': 
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim_name.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            print(f"{optim_name} not implemented")
            raise NotImplementedError

    def generate_scheduler(self, optimizer):
        '''
        return torch.optim.lr_scheduler
        '''
        scheduler_name = self.cfg.optim.scheduler.name
        if scheduler_name.lower() == 'CosineAnnealingLR'.lower():
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.optim.scheduler.t_max)
        elif scheduler_name.lower() == 'StepLR'.lower():
            step_size = self.cfg.optim.scheduler.step_size
            gamma = self.cfg.optim.scheduler.gamma
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == 'MultiStepLR'.lower():
            milestones = self.cfg.optim.scheduler.milestones
            gamma = self.cfg.optim.scheduler.gamma
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
            mode = self.cfg.optim.scheduler.mode
            patience = self.cfg.optim.scheduler.patience
            verbose = self.cfg.optim.scheduler.verbose
            factor = self.cfg.optim.scheduler.gamma
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, 
                                                        verbose=verbose, factor=factor)
        else:
            print(f"{scheduler_name} not implemented")
            raise NotImplementedError

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = self.generate_optimizer()
        scheduler = self.generate_scheduler(optimizer)
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
