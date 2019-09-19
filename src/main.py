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

import argparse

parser = argparse.ArgumentParser()


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

# SGD
def sigmoid(x):
    '''
    Sigmoid
    
    Attributes:
        * x (???)
    '''
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    '''
    Sigmoid derivative
    
    Attributes:
        * x (???)
    '''
    return x * (1.0 - x)

def MSE(predicted_val, true_val):
    '''
    Mean Error Squared
    
    Attributes:
        * predicted_val (numpy array): Predicted value
        * true_val (numpy array): Correct value
    '''
    return np.square(np.subtract(true_val,predicted_val)).mean() 
    
# initialization
params = {"W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
          "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
          "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
          "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h)}


