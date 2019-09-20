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

# Loads the data
mnist_data = pickle.load(open('../data/mnist.pkl', 'rb'))
x_train = mnist_data['training_images']
y_train = mnist_data['training_labels']
x_test  = mnist_data['test_images']
y_test  = mnist_data['test_labels']


# HYPER PARAMETERS
digits = 10
n_x = 784
x_h = 64
n_h = 64
learn_rate = 0.5
epochs = 50

#%%

# Init
weights1 = np.random.rand(n_x, 16)
weights2 = np.random.randn(16, 16)
weights3 = np.random.randn(16, 10)

bias1 = np.random.randn(1, 16)
bias2 = np.random.randn(1, 16)

# Selecting the data
i = 0
selected_data = i
input_data = x_train[selected_data]
correct_answer = y_train[selected_data]

# Feedforward
layer1 = sigmoid(np.dot(input_data, weights1) + bias1)
layer2 = sigmoid(np.dot(layer1, weights2) + bias2)
output = sigmoid(np.dot(layer2, weights3)) 

cost = MSE(output, correct_answer)
print("Cost:", cost)