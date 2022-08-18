# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:51:44 2020

@author: laukkara
"""

from sklearn.base import BaseEstimator

import numpy as np
from scipy.optimize import curve_fit


class ExpFuncRegressor(BaseEstimator):
    
    def __init__(self, a=-1.0, b=0.6, c=0.8, d=0.1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    
    def fit(self, X, y):
        # Incoming X and y and 2D arrays
        # Default initial values for parameters in curve_fit are not
        # good for this data and cause np.inf -values.
        # Custom lower and upper bounds are provided, which fixes it.
        
        p0 = [self.a, self.b, self.c, self.d]
        lb = [-2, 0.0, 0.0, -2.0]
        ub = [1, 2.0, 2.0, 2.0]
        
        
        popt, pcov = curve_fit(self.exp_func, 
                               X[:,0], y[:,0],
                               bounds=(lb, ub),
                               p0=p0)
        self.a = popt[0]
        self.b = popt[1]
        self.c = popt[2]
        self.d = popt[3]
        # print('fit end', self.a, self.b, self.c, self.d)
        return(self)
    
    
    def predict(self, X):
        # Incoming X is 2D array
        #print('myClasses predict', X.shape)
        y_pred = self.exp_func(X[:,0], self.a, self.b, self.c, self.d)
        return(y_pred)
    
    
    @staticmethod
    def exp_func(x, a, b, c, d):
        #print('staticmethod start', x.shape)
        y = a + b*np.exp(c*(x-d))
        #print('staticmethod end', x.shape)
        return(y)
    
    
    
class PiecewiseRegressor(BaseEstimator):
    
    def __init__(self, a=10.0, b=20.0, k1=0.0, k2=1.0):
        self.a = a
        self.b = b
        self.k1 = k1
        self.k2 = k2
    
    
    def fit(self, X, y):
        # Incoming X and y and 2D arrays
        # Default initial values for parameters in curve_fit are not
        # good for this data and cause np.inf -values.
        # Custom lower and upper bounds are provided, which fixes it.
        
        p0 = [self.a, self.b, self.k1, self.k2]
        lb = [10.0,  15.0, -0.5, 0.0]
        ub = [20.0, 25.0,  0.5, 2.0]
        
        
        popt, pcov = curve_fit(self.piecewise_func, 
                               X[:,0], y[:,0],
                               bounds=(lb, ub),
                               p0=p0)
        self.a = popt[0]
        self.b = popt[1]
        self.k1 = popt[2]
        self.k2 = popt[3]
        # print('fit end', self.a, self.b, self.c, self.d)
        return(self)
    
    
    def predict(self, X):
        # Incoming X is 2D array
        #print('myClasses predict', X.shape)
        y_pred = self.piecewise_func(X[:,0], self.a, self.b, self.k1, self.k2)
        return(y_pred)
    
    
    @staticmethod
    def piecewise_func(x, a, b, k1, k2):
        y = np.zeros(len(x))
        for idx, val in enumerate(x):
            if val < a:
                # linear function 1
                y[idx] = k1*(val-a) + b
            else:
                # linear function 2
                y[idx] = k2*(val-a) + b
        return(y)
    
    
    
