#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:48:53 2019

@author: carlelg
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from functions import *


# HYPER PARAMETERS
digits = 10
n_x = 784
x_h = 64
n_h = 64
beta = 0.9
lr = 0.5
epochs = 50

# Loads the data
mnist_data = pickle.load(open('../data/mnist.pkl', 'rb'))
x_train = mnist_data['training_images']
y_train = mnist_data['training_labels']
x_test  = mnist_data['test_images']
y_test  = mnist_data['test_labels']



