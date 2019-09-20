#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from logging.config import dictConfig as _dictConfig
from os import path

import yaml

__author__ = 'Somshubra Majumdar'
__version__ = '0.0.2'


def get_logger(name=None):
    with open(path.join(path.dirname(__file__), '_data', 'logging.yml'), 'rt') as f:
        data = yaml.load(f)
    _dictConfig(data)
    return logging.getLogger(name=name)


root_logger = get_logger()

try:
    from tensorflow import __version__ as tf_version

    TF = True
except ImportError:
    TF = False
