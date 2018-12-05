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

# First picture
input_data = mnist_data['training_images'][0]


# Activation functions
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)


#%%
### INIT ###

# Hyper parameters
# 728 input, 28x28 picture
# 2@16 Hidden neuron layers
# 10 cells as output, 10 digits

# Biases
b1 = np.random.randn(16, 1)
b2 = np.random.randn(16, 1)
b3 = np.random.randn(16, 1)
b_output = np.random.randn(10, 1)

# Weights
w1 = np.random.randn(input_data.shape[0], 1)
w2 = np.random.randn(16, 1)
w3 = np.random.randn(16, 1)
w_output = np.random.randn(10, 16)

## Feedforward ##
layer1 = sigmoid(np.dot(input_data, w1) + b1)
layer2 = sigmoid(np.dot(layer1.transpose(), w2) + b2)
layer3 = sigmoid(np.dot(layer2.transpose(), w3) + b3)

output = sigmoid(np.dot(w_output, layer3) + b_output)


## ERROR  ##

def MSE(output):

















