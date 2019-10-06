#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Somshubra Majumdar'
__version__ = '0.0.4'

try:
    from tensorflow import __version__ as tf_version

    TF = True
except ImportError:
    TF = False
