#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import time
import os

__all__ = [
    'Logger'
]

class Logger(object):
    def __init__(self, logger_name=None, cfg=None):
        '''
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        '''

        # 创建一个logger
        if cfg is None:
            file = 'log.txt'
        else:
            file = cfg.log.name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        formatter = logging.Formatter('[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')

        if file:
            hdlr = logging.FileHandler(file, 'a', encoding='utf-8')
            hdlr.setLevel(logging.INFO)
            hdlr.setFormatter(formatter)
            self.logger.addHandler(hdlr)

        strhdlr = logging.StreamHandler()
        strhdlr.setLevel(logging.INFO)
        strhdlr.setFormatter(formatter)
        self.logger.addHandler(strhdlr)

        if file: hdlr.close()
        strhdlr.close()

    def getlogger(self):
        return self.logger
