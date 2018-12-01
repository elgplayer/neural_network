# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:31:10 2018

@author: Carl
"""

import os
import pickle

import numpy as np


# Loads the data
mnist_data = pickle.load(open('../data/mnist.pkl', 'rb'))


# Activation functions
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


#%%
## INIT ##

# Hyper parameters
sizes = [784, 16, 16, 10]

# 728 input, 28x28 picture
# 2@16 Hidden neuron layers
# 10 cells as output, 10 digits

num_layers = len(sizes)

# Biases
b = [np.random.randn(y, 1) for y in sizes[1:]]

# Weights
w = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]




