# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:20:04 2020

@author: ayanca
"""

import warnings
import numpy as np
import pandas as pd
from ML_methods_diabetes import *
from normality_test import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

################################################################################################################################################
#load data
path = r"C:\Users\ayanca\.spyder-py3\obesity_paper_1\diabetes\pima-indians-diabetes-database\diabetes.csv"

headernames = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
final_column = ['Glucose', 'BloodPressure', 'BMI','DiabetesPedigreeFunction','Age','Outcome']

data = pd.read_csv(path, na_values="?", low_memory=False)
data = data[final_column]

warnings.filterwarnings('ignore')
np.seterr(divide = 'ignore') 
np.seterr(all = 'ignore') 

print (data.shape)

#check if missing value (cleaning)
data = remove_missing(data)

check_feature(data)

corr_sort = find_feature_reduced_matrix(data.corr())
print ("Strong dependency,", corr_sort)

to_drop = find_feature_reduced_matrix(data.corr())
print (data.columns)
data.drop(to_drop, axis=1, inplace=True)

data_visualize(data, final_column)
print (data.columns)



features = data.shape[1]-1
print (features)
print (data.head(10))

#remove duplicate
if data.duplicated().sum()>0:
    print ("found and removed...",data.duplicated().sum())
    data.drop_duplicates(inplace = True)
 
#filtering age
data = data[data['Age']>20]
data = data[data['Age']<60]

################################################################################################################################################
data = data.values
data = np.array(data, dtype = float)

#############column selection based on p-value###############
selected_columns = data
'''
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
print (data_modeled)
print (selected_columns)
'''
##########################ends###############################

#shuffle data
np.random.shuffle(selected_columns) 
print ("Data shape after shuffling:", selected_columns.shape)

################################################################################################################################################
#ML-Data Ready
fold = 5
X, y = data[:, :features], data[:,features]
#print (X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=33)

#Individual Model
#ML-SVM
'''
print ("SVM............linear........")
svm_linear_accuracy(X_train, X_test, y_train, y_test, fold)
print ("LR............logistic regression........")
lr_accuracy(X_train, X_test, y_train, y_test, fold)
print ("NB..........GNB..........")
NB_accuracy_gaussian(X_train, X_test, y_train, y_test, fold)
'''

#ML-compare
print ("Compare Machine Learning Algorithms Consistently....................")
print ("5 fold cross validation...")
#ensemble_all_general(X, y, fold = 5)
plot_ml_model(X, y, fold = 5)
#print ("Gid search 5 fold")
#grid_search(X, y, fold)

#ensemble_voting_classifier(X_train, X_test, y_train, y_test, fold)

calibration_model_compare (X_train, y_train, X_test, y_test)
calibration(X_train, y_train, X_test, y_test,GaussianNB(),"Naive Bayes",2)

#calibration(X_train, y_train, X_test, y_test, LogisticRegression(),"LR",2)
#from sklearn.tree import DecisionTreeClassifier
#calibration(X_train, y_train, X_test, y_test,DecisionTreeClassifier(),"DTree",2)