#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Sep 19 10:48:53 2019

@author: carlelg
'''

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from functions import *

# Loads the data
mnist_data = pickle.load(open('../data/mnist.pkl', 'rb'))
x_train = mnist_data['training_images']
y_train = mnist_data['training_labels'].reshape(-1, 1)
x_test  = mnist_data['test_images']
y_test  = mnist_data['test_labels'].reshape(-1, 1)


# HYPER PARAMETERS
n_x = 784 # Input
n_h = 64 # Layer size of the deep neurons


### Create One hot encoding ###
X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))

digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

m = 60000
m_test = X.shape[0] - m
X_train = X[:m].T
X_test = X[m:].T
Y_train = Y_new[:, :m]
Y_test = Y_new[:, m:]

# Shuffle training set
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]


#%%


# Hmm redo?
params = {
    'W1': np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
    'b1': np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
    'W2': np.random.randn(n_h, digits) * np.sqrt(1. / n_h),
    'b2': np.zeros((digits, 1)) * np.sqrt(1. / n_h)
    }


def feed_forward(X, params):
    
    tmp = {}
    
    tmp['z_1'] = np.matmul(params["W1"], X) + params["b1"]
    tmp['layer_1'] = sigmoid(tmp['z_1'])
    
    tmp['z_2'] = np.matmul(params["W2"], X) + params["b2"]
    tmp['layer_2'] = sigmoid(tmp['z_2'])
    
    return tmp    









#%%
# HYPER PARAMETERS
#digits = 10
#n_x = 784
#x_h = 64
#n_h = 64
#learn_rate = 0.5
#epochs = 50

#%%

## Init
#weights1 = np.random.rand(n_x, 16)
#weights2 = np.random.rand(16, 16)
#weights3 = np.random.rand(16, 10)
#
#bias1 = np.random.rand(1, 16)
#bias2 = np.random.rand(1, 16)
#
## Selecting the data
#i = 0
#selected_data = i
#input_data = x_train[selected_data]
#correct_answer = y_train[selected_data]
#
## Feedforward
#layer1 = sigmoid(np.dot(input_data, weights1) + bias1)
#layer2 = sigmoid(np.dot(layer1, weights2) + bias2)
#output = sigmoid(np.dot(layer2, weights3)) 
#
## Nice print
#pred_value = np.argmax(output)
#is_correct = 'Wrong'
#if pred_value == correct_answer:
#    is_correct = 'Correct'
#print('{}! | Predicted Value: {} | Actual value: {}'.format(is_correct, pred_value, 
#      correct_answer))
#
## Backprop
#output_error = MSE(output, correct_answer)
#output_delta = output_error * sigmoid_derivative(output)
#
#z3_cost = output_delta.dot(weights3.T)
