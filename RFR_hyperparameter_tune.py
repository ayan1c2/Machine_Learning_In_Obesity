# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 00:04:00 2020

@author: ayanca
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))    
    return accuracy


# Create the parameter grid based on the results of random search 
def RFR_tune(X_train, X_test, y_train, y_test,fold):
    train_features = X_train
    train_labels = y_train

    param_grid = {
    'bootstrap': [True],
    'max_depth': [50, 80, 90, 100, 110, 150],
    'max_features': [2, 3, 5, 7],
    'min_samples_leaf': [3, 4, 5, 6],
    'min_samples_split': [8, 10, 12, 14, 16],
    'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = fold, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(train_features, train_labels)
    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    print (best_grid)
    grid_accuracy = evaluate(best_grid, X_test, y_test)
    print (grid_accuracy)