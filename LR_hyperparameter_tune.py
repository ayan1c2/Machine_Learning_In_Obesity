# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:32:59 2020

@author: ayanca
"""

# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

def RFR_tune(X,y,fold):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Create logistic regression
    logistic = linear_model.LogisticRegression()
    
    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)
    
    # Create grid search using 5-fold cross validation
    clf = GridSearchCV(logistic, hyperparameters, cv=fold, verbose=0)
    
    # Fit grid search
    best_model = clf.fit(X, y)
    
    # View best hyperparameters
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    
    # Predict target vector
    best_model.predict(X)
    
#RFR_tune(3,3,5)