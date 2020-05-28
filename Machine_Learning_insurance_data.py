# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:34:44 2020

@author: ayanca
"""

################################################################################################################################################
#load library
import warnings
import numpy as np
import pandas as pd
from ML_methods_insurance import *
from normality_test import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols
from RFR_hyperparameter_tune import RFR_tune

################################################################################################################################################
#load data
path = r"C:\Users\ayanca\.spyder-py3\obesity_paper_1\insurance\insurance.csv"

headernames = ['age', 'sex', 'bmi', 'children','smoker','region','charges']
#coln_classification = ['age', 'sex', 'bmi', 'children','smoker','region','body_composition']
#coln_regression = ['age', 'sex', 'bmi', 'smoker','region','charges']
feature_cols = ['age', 'sex', 'bmi','smoker','region']
total_col = ['age', 'sex', 'bmi', 'smoker','region','charges', 'body_composition']

data = pd.read_csv(path, na_values="?", low_memory=False)

warnings.filterwarnings('ignore')
np.seterr(divide = 'ignore') 
np.seterr(all = 'ignore') 

################################################################################################################################################
#encoding
lb_make = LabelEncoder()
data["sex"] = lb_make.fit_transform(data["sex"])
data["smoker"] = lb_make.fit_transform(data["smoker"])
data["region"] = lb_make.fit_transform(data["region"])

print (data.shape)

#check if missing value (cleaning)
data = remove_missing(data)
'''
"Underweight" = 0
"Normal Weight" = 1
"Overweight" = 2
"Obese" = 3
'''
data['body_composition'] = data.apply(calculate_body_composition, axis=1)
data = data[total_col]

#############for multiclass to binary conversion################
#data['body_composition'] = data.apply(convert_status_to_description, axis=1)
#data = data[total_col]
#print (data.head())

#check_feature(data)

corr_sort = find_feature_reduced_matrix(data.corr())
print ("Strong dependency,", corr_sort)

to_drop = find_feature_reduced_matrix(data.corr())
print ("Remove dependency,", to_drop)
data.drop(to_drop, axis=1, inplace=True)

print (data.columns)

features = data.shape[1]-1
print (data.head(10))

#data.to_csv(r'export_dataframe_insurance.csv', index = False, header=True)

data_visualize(data, total_col)
data = data.drop(['charges'], axis=1) #for correlation
#data = data.drop(['body_composition'], axis=1) #for regression
print (data.columns)

features = data.shape[1]-1
print (data.head(10))


#bmi_plot(data[data['bmi']>=25])

#P-Value Anova
#moore_lm = ols("bmi ~ age_cat", data=data).fit()
#print(moore_lm.summary())

#remove duplicate
if data.duplicated().sum()>0:
    print ("found and removed...",data.duplicated().sum())
    data.drop_duplicates(inplace = True)

'''
#check distribution of data
print ("Whole data distribution: ",shapiro_normality_test(data))

for (columnName, columnData) in data.iteritems():
   print('Colunm Name : ', columnName)
   print('Column distribution : ', shapiro_normality_test(columnData.values))
 ''' 

#filtering age
data = data[data['age']>20]
data = data[data['age']<60]
 
################################################################################################################################################
################################################################################################################################################
data = data.values
data = np.array(data, dtype = float)

#############column selection based on p-value###############
selected_columns = data

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

##########################ends###############################

################################################################################################################################################
#ML-Data Ready

fold = 5
X, y = data[:, :features], data[:,features]
#print (X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=33)

#ML-compare
print ("Compare Machine Learning Algorithms Consistently....................")
print ("5 fold cross validation...")
#ensemble_all_general(X, y, fold = 5)
plot_ml_model(X, y, fold = 5)

#print ("10 fold cross validation...")
#ensemble_all_general(X, y, fold = 10)
#plot_ml_model(X, y, fold = 10)

#print("Grid search fold=5")
#grid_search(X,y,fold=5)
#print("Grid search fold=10")
#grid_search(X,y,fold=10)

##############################################Regression###############################
#ML-compare
#print ("5 fold cross validation...")
#ensemble_all_general(X, y, fold = 5)
#plot_ml_model_regression(X_train, X_test, y_train, y_test)

#print ("10 fold cross validation...")
#ensemble_all_general(X, y, fold = 10)
#plot_ml_model_regression(X, y, fold = 10)

#DT_tune(X,y,fold)
#RFR_tune(X_train, X_test, y_train, y_test,fold)

#ML-compare
#print ("Voting Ensemble Algorithms....................")
#ensemble_voting_classifier(X_train, X_test, y_train, y_test, fold)

#RF_regressor_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)
################################################################################################################################################
#ML-DT-gini
#print ("DT.......gini.............")
#DT_gini_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)

################################################################################################################################################
#ML-DT-entropy
#print ("DT.......entropy.............")
#DT_entropy_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)


#######################################################binary classification#########################################################
#binary classification
#ML-compare
#print ("Compare Machine Learning Algorithms Consistently....................")
#print ("5 fold cross validation...")
#ensemble_all_general(X, y, fold = 5)
#plot_ml_model(X, y, fold = 5) 

#calibration_model_compare (X_train, y_train, X_test, y_test)
#from sklearn.tree import DecisionTreeClassifier
#calibration(X_train, y_train, X_test, y_test,DecisionTreeClassifier(),"DTree",2)
#from sklearn.tree import DecisionTreeClassifier
#calibration(X_train, y_train, X_test, y_test,DecisionTreeClassifier(),"DTree",2)