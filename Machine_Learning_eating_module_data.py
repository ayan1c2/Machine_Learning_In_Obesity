# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:10:19 2020

@author: ayanca
"""

#load library
import warnings
import numpy as np
import pandas as pd
from ML_methods_eating_module import *
from normality_test import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols
from DT_hyperparameter_tune import DT_tune

################################################################################################################################################
#load data
path = r"C:\Users\ayanca\.spyder-py3\obesity_paper_1\eating-health-module-dataset\ehresp_2014.csv"

headernames = ['erbmi','eusoda', 'eusnap', 'euincome2', 'eugenhth', 'erincome', 'eudietsoda',
                  'euffyday', 'eufdsit', 'eufastfdfrq','ertseat', 'eudrink', 'eueat',
                  'euexfreq', 'euexercise', 'eufastfd', 'eumeat', 'eumilk', 'eustores', 
                  'eustreason', 'euwic']

total_col = ['erbmi','eusoda', 'eusnap', 'eugenhth', 'euincome2', 'eudietsoda',
                  'euffyday', 'eufdsit', 'eufastfdfrq','ertseat', 'eudrink', 'eueat',
                  'euexfreq', 'euexercise', 'eufastfd', 'eumeat', 'eumilk', 'eustores', 
                  'eustreason', 'euwic', 'body_composition']

feature_cols = ['erbmi','eusoda', 'eusnap', 'eugenhth', 'euincome2', 'eudietsoda',
                  'euffyday', 'eufdsit', 'eufastfdfrq','ertseat', 'eudrink', 'eueat',
                  'euexfreq', 'euexercise', 'eufastfd', 'eumeat', 'eumilk', 'eustores', 
                  'eustreason', 'euwic']

data = pd.read_csv(path, na_values="?", low_memory=False)
print(data.head(), data.shape)

warnings.filterwarnings('ignore')
np.seterr(divide = 'ignore') 
np.seterr(all = 'ignore') 
data = remove_missing(data)
################################################################################################################################################
data = data [data>-1]
data = data[headernames]
print (data)

'''
nan_value = float("NaN")
data.replace("", nan_value, inplace=True)
data.dropna(inplace=True)
print(data.columns)
'''

data = remove_missing(data)

#data_visualize(data, headernames)

corr_sort = find_feature_reduced_matrix(data.corr())
print ("Strong dependency,", corr_sort)
to_drop = find_feature_reduced_matrix(data.corr())
print ("Remove dependency,", to_drop)
data.drop(to_drop, axis=1, inplace=True)

'''
"Underweight" = 0
"Normal Weight" = 1
"Overweight" = 2
"Obese" = 3
'''
data['body_composition'] = data.apply(calculate_body_composition, axis=1)
#data['body_composition'] = data.apply(convert_status_to_description, axis=1) #for binary classification
data = data[total_col]
print (data.head(), data.shape)

#check_feature(data)

features = data.shape[1]-1
print (data.head(10))

#remove duplicate
if data.duplicated().sum()>0:
    print ("found and removed...",data.duplicated().sum())
    data.drop_duplicates(inplace = True)
    
#############column selection based on p-value###############
selected_columns = data.values
'''
#for multiclass classification only, comment it for binary classification
import statsmodels.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
p_val = 0.05
data_modeled, selected_columns = backwardElimination(data[:, :features], data[:,features], p_val, selected_columns)
#print (data_modeled)
'''
##########################ends###############################

#shuffle data
np.random.shuffle(selected_columns) 
print ("Data shape after shuffling:", selected_columns.shape)

fold = 5
X, y = selected_columns[:, :features], selected_columns[:,features]
#print (X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=33)

#ML-compare
print ("Compare Machine Learning Algorithms Consistently....................")
print ("5 fold cross validation...")
#ensemble_all_general(X, y, fold = 5)
plot_ml_model(X, y, fold)

################################################################################################################################################
#ML-DT-gini
#print ("DT.......gini.............")
#DT_gini_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)

################################################################################################################################################
#ML-DT-entropy
#print ("DT.......entropy.............")
#DT_entropy_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)


#DT_tune(X,y,fold)

#ensemble_voting_classifier(X_train, X_test, y_train, y_test, fold)

#######################################################binary classification#########################################################
#binary classification
#ML-compare
#print ("Compare Machine Learning Algorithms Consistently....................")
#print ("5 fold cross validation...")
#ensemble_all_general(X, y, fold = 5)
#plot_ml_model(X, y, fold = 5) #SVM, DTree,RF are the best models

#calibration_model_compare (X_train, y_train, X_test, y_test)
#calibration(X_train, y_train, X_test, y_test,SVC(kernel='linear'),"SVC",2)
#from sklearn.tree import DecisionTreeClassifier
#calibration(X_train, y_train, X_test, y_test,DecisionTreeClassifier(),"DTree",2)