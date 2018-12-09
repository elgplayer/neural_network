# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:31:10 2018

@author: Carl
"""

import os
import pickle

import numpy as np

### INIT ###

# Loads the data
mnist_data = pickle.load(open('../data/mnist.pkl', 'rb'))


# Activation functions
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - sigmoid(x))


# Hyper parameters
# 728 input, 28x28 picture
# 2@16 Hidden neuron layers
# 10 cells as output, 10 digits

learning_rate = 0.1

# Weights  
w2 = np.random.randn(16, 784)
w3 = np.random.randn(16, 16)
w4 = np.random.randn(10, 16)

# Biases
b2 = np.random.randn(16, 1)
b3 = np.random.randn(16, 1)
b4 = np.random.randn(10, 1)
#%%

# Loopy loop
for k in range(100):
    
    selected_data = k
    input_data = mnist_data['training_images'][selected_data]
    input_data = input_data.reshape(784, 1)
    correct_answer = mnist_data['training_labels'][selected_data]

    ## Feedforward ##
    z2 = np.matmul(w2, input_data) + b2
    layer2 = sigmoid(z2)
    
    z3 = np.matmul(w3, layer2) + b3
    layer3 = sigmoid(z3)
    
    z4 = np.matmul(w4, layer3) + b4
    layer4 = sigmoid(z4)
    
    output = layer4
    #%%
    
    # Error
    cost = 0
    
    for i in range(len(output)):
        
        if i == correct_answer:
            
            cost += (output[i][0] - 1)**2
        
        else: 
            
            cost += (output[i][0] - 0)**2
    
    # Nabla C
    nabla_c = np.zeros((10, 1))
            
    for i in range(len(output)):
        
        if i == correct_answer:
            
            nabla_c[i] = (1 - output[i][0])
        
        else: 
            
            nabla_c[i] = (0 - output[i][0])
            
    
    #%%
    # Output error 
    nabla_4 = np.multiply(nabla_c, sigmoid_derivative(z4))
    
    nabla_3 = np.multiply(np.matmul(w4.transpose(), nabla_4), sigmoid_derivative(z3))
    
    nabla_2 = np.multiply(np.matmul(w3.transpose(), nabla_3), sigmoid_derivative(z2))
    

    
    #%%
    ## Gradient decent ##
    
    # Weights update
    w4 = w4 - learning_rate * np.matmul(nabla_4, layer3.transpose())
    
    w3 = w3 - learning_rate * np.matmul(nabla_3, layer2.transpose())
    
    w2 = w2 - learning_rate * np.matmul(nabla_2, input_data.transpose())
    
    # Biases update
    b4 = b4 - learning_rate * nabla_4
    
    b3 = b3 - learning_rate * nabla_3
    
    b2 = b2 - learning_rate * nabla_2

    #%%
    
    print(k, cost)


