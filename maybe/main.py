# -*- coding: utf-8 -*-

#from sklearn.datasets import fetch_mldata


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

from functions import *

mnist_data = pickle.load(open('../data/mnist.pkl', 'rb'))
x_train = mnist_data['training_images']
y_train = mnist_data['training_labels']
x_test  = mnist_data['test_images']
y_test  = mnist_data['test_labels']

X = x_train
X = X / 255
y = y_train

y_new = np.zeros(y.shape)
y_new[np.where(y == 0.0)[0]] = 1
y = y_new

m = 60000
m_test = 10000

X_train = X.T
X_test = (x_test / 255).T
y_train = y.reshape(1,m)
y_test = y_test.reshape(1,m_test)

np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

#%%

#m = 60000 # Number of training data
#m_test = 10000 # Number of test data
#np.random.seed(138) # Set the random seed
#
## Normalize the data
#x_train = (x_train / 255).T
#x_test = (x_test /  255).T
#
#
#y_train_new = np.zeros(y_train.shape)
#y_train_new[np.where(y_train == 0.0)[0]] = 1
#y_train = y_train_new
#y_train = y_train.reshape(1, m)
#
#y_test_new = np.zeros(y_test.shape)
#y_test_new[np.where(y_test == 0.0)[0]] = 1
#y_test = y_test_new
#y_test = y_test.reshape(1, m_test)
#
## Shuffle index
#shuffle_index = np.random.permutation(m)
#x_train, y_train = x_train[:,shuffle_index], y_train[:,shuffle_index]

#%%
#learning_rate = 1
#
#X = X_train
#Y = y_train
#
#n_x = X.shape[0]
#m = X.shape[1]
#
#W = np.random.randn(n_x, 1) * 0.01
#b = np.zeros((1, 1))


X = X_train
Y = y_train

n_x = X.shape[0]
n_h = 64
learning_rate = 1

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(1, n_h)
b2 = np.zeros((1, 1))




#%%

for i in range(2000):
    
    print(i)
    
    Z1 = np.matmul(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)

    cost = compute_loss(Y, A2)
    break

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    
    

    if i % 100 == 0:
        print("Epoch", i, "cost: ", cost)
        #break

print("Final cost:", cost)