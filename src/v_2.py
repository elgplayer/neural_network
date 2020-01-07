# -*- coding: utf-8 -*-

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

# Hyper parameters
n_h = 64 # Depp neuron size
n_x = 784 # Input size
digits = 10 # Number of output

#np.random.seed(500)

params = {
    'w_1': np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
    'b_1': np.random.randn(n_h, 1),
    'w_2': np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
    #'b_2': np.zeros((digits, 1)) * np.sqrt(1. / n_h)
    'b_2': np.random.randn(digits, 1)
    }


#def feedforward(x_input, params):
x_input = x_train[0]    

tmp = {}

# Dot product
tmp['z_1'] = np.dot(params['w_1'], x_input) + params['b_1']
tmp['layer_1'] = sigmoid(tmp['z_1']) #Sigmoid squishification

#tmp['z_2'] = np.dot(params['w_2'], tmp['layer_1']) + params['b_2']
tmp['z_2'] = np.sum(np.dot(params['w_2'], tmp['layer_1']), axis=1) #+ params['b_2']
tmp['layer_2'] = sigmoid(tmp['z_2']) #Sigmoid squishification
#tmp['layer_2'] = np.exp(tmp["z_2"]) / np.sum(np.exp(tmp["z_2"]), axis=0)

