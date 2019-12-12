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
from torchvision.datasets import MNIST

from torchline.data import build_data
from torchline.losses import build_loss_fn
from torchline.models import build_model


class LightningTemplateModel(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, cfg):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param cfg:
        """
        # init superclass
        super(LightningTemplateModel, self).__init__()
        self.cfg = cfg

        self.batch_size = self.cfg.DATASET.BATCH_SIZE

        # if you specify an example input, the summary will show input/output for each layer
        h, w = self.cfg.INPUT.SIZE
        self.example_input_array = torch.rand(1, 3, h, w)

        # build model
        self.model = self.build_model(cfg)
        self.loss_fn = self.build_loss_fn(cfg)

    # ---------------------
    # MODEL SETUP
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
        cfg.DATASET.IS_TRAIN = is_train
        cfg.freeze()
        return build_data(cfg)



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
        return self.loss_fn(predictions, gt_labels)

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
        predict_labels = torch.argmax(predictions, dim=1)
        train_acc = torch.sum(gt_labels== predict_labels).item() / (len(gt_labels) * 1.0)
        train_acc = torch.tensor(train_acc)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val, 'train_acc': train_acc}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })


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
        predict_labels = torch.argmax(predictions, dim=1)
        val_acc = torch.sum(gt_labels== predict_labels).item() / (len(gt_labels) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

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
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def generate_optimizer(self):
        '''
        return torch.optim.Optimizer
        '''
        optim_name = self.cfg.OPTIM.NAME
        momentum = self.cfg.OPTIM.MOMENTUM
        weight_decay = self.cfg.OPTIM.WEIGHT_DECAY
        if optim_name.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.cfg.OPTIM.BASE_LR, 
                                        momentum=momentum, weight_decay=weight_decay)
        elif optim_name.lower() == 'adadelta':
            return torch.optim.Adagrad(self.parameters(), lr=self.cfg.OPTIM.BASE_LR, 
                                            weight_decay=weight_decay)
        elif optim_name.lower() == 'adam': 
            return torch.optim.Adam(self.parameters(), lr=self.cfg.OPTIM.BASE_LR, 
                                        weight_decay=weight_decay)
        elif optim_name.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.cfg.OPTIM.BASE_LR, 
                                            momentum=momentum, weight_decay=weight_decay)
        else:
            print(f"{optim_name} not implemented")
            raise NotImplementedError

    def generate_scheduler(self, optimizer):
        '''
        return torch.optim.lr_scheduler
        '''
        scheduler_name = self.cfg.OPTIM.SCHEDULER.NAME
        if scheduler_name.lower() == 'CosineAnnealingLR'.lower():
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        elif scheduler_name.lower() == 'StepLR'.lower():
            step_size = self.cfg.OPTIM.SCHEDULER.STEP_SIZE
            gamma = self.cfg.OPTIM.SCHEDULER.GAMMA
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == 'MultiStepLR'.lower():
            milestones = self.cfg.OPTIM.SCHEDULER.MILESTONES
            gamma = self.cfg.OPTIM.SCHEDULER.GAMMA
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
            mode = self.cfg.OPTIM.SCHEDULER.MODE
            patience = self.cfg.OPTIM.SCHEDULER.PATIENCE
            verbose = self.cfg.OPTIM.SCHEDULER.VERBOSE
            factor = self.cfg.OPTIM.SCHEDULER.GAMMA
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
        train_sampler = None
        batch_size = self.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS
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
