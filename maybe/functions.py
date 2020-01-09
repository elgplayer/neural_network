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
    

def compute_loss(Y, Y_hat):

    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )

    return L


def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L


def picture_plot(x):
    '''
    Plots the picture given a 784 Numpy array
    
    Attributes:
        x (numpy array): Pixel values
    '''
    
    img = x.reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    

def weight_init(n):
    
    return np.random.randn(n) / np.sqrt(n)
    
