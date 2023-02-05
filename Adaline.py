# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:37:05 2023

@author: Tim
"""

import numpy as np

class AdalineGD(object):
    """ADAptive LInear NEuron classifier from Raschka's Python Machine Learning
    
    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    num_iter : int
        Max iterations over training set.
    rand_state : int
        random number generator seed for initial weights.
        
    Attributes
    ----------
    weight : 1d-array
        Vector of fitted weights.
    cost : list
        Sum-of-squares cost function after every epoch.
    
    """
    def __init__(self, eta = 0.01, num_iter = 25, rand_state = 3):
        self.eta = eta
        self.num_iter = num_iter
        self.rand_state = rand_state
        
    def fit(self, X, y):
        """Fit training data
    
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is number of examples
            and n_features is number of features.
        y : array-like, shape = [n_examples]
            target values
            
        Returns
        -------
        self : object
        
        """
        rand = np.random.RandomState(self.rand_state)
        self.weight = rand.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        
        self.cost = []
        
        for i in range(self.num_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.weight[1:] += self.eta * X.T.dot(errors)
            self.weight[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input (training vector and weight products)"""
        return np.dot(X, self.weight[1:]) + self.weight[0]
    
    def activation(self, X):
        """Computes linear activation """
        return X
    
    def predict(self, X):
        """Predicts class label after unit step."""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    

