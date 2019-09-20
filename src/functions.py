# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


# SGD
def sigmoid(x):
    '''
    Sigmoid
    
    Attributes:
        * x (???)
        
    returns: Sigmoid value
    '''
    
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    '''
    Sigmoid derivative
    
    Attributes:
        * x (???)
        
    returns: Sigmoid derative value
    '''
    return x * (1.0 - x)


def MSE(predicted_val, true_val):
    '''
    Mean Error Squared
    
    Attributes:
        * predicted_val (numpy array): Predicted value
        * true_val (numpy array): Correct value
        
    returns: Cost
    '''
    
    return np.square(np.subtract(true_val,predicted_val)).mean() 
    

def picture_plot(x):
    '''
    Plots the picture given a 784 Numpy array
    
    Attributes:
        x (numpy array): Pixel values
    '''
    
    pixels = x.reshape((28, 28))
    plt.imshow(pixels)
    
