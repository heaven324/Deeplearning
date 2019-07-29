# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:02:59 2019

@author: heaven
"""

import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x