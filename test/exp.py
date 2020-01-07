# -*- coding: utf-8 -*-

def backpropagation(self,y, z_s, a_s):

  dw = []  # dC/dW
  db = []  # dC/dB

  deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer

  # insert the last layer error
  deltas[-1] = ((y-a_s[-1])*(self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1]))

  # Perform BackPropagation
  for i in reversed(range(len(deltas)-1)):
    
    deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))        
    batch_size = y.shape[1]
    db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
    dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
    # return the derivitives respect to weight matrix and biases
    return dw, db