# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:13:27 2020

@author: ayanca
"""

################################################################################################################################################
#load library
import warnings
import numpy as np
import pandas as pd
from ML_methods import *
from normality_test import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

################################################################################################################################################
#load data
path = r"C:\Users\ayanca\.spyder-py3\obesity_paper_1\BMI\BMI_Person_Gender_Height_Weight_Index.csv"
headernames = ['Gender', 'Height', 'Weight', 'Index']
#coln_binary = ['Gender', 'Height', 'Weight', 'BMI', 'Index','Risk'] #for binary classification
coln = ['Gender', 'Height', 'Weight', 'BMI', 'Index'] #for multiclass classification
feature_cols = ['Gender', 'Height', 'Weight', 'BMI']
data = pd.read_csv(path, na_values="?", low_memory=False)

warnings.filterwarnings('ignore')
np.seterr(divide = 'ignore') 
np.seterr(all = 'ignore') 

################################################################################################################################################
#encoding
lb_make = LabelEncoder()
data["Gender"] = lb_make.fit_transform(data["Gender"])

#check if missing value (cleaning)
data = remove_missing(data)
#print (pd.unique(data["Index"]))

#check distribution of data
print ("Whole data distribution: ",shapiro_normality_test(data))

for (columnName, columnData) in data.iteritems():
   print('Colunm Name : ', columnName)
   print('Column distribution : ', shapiro_normality_test(columnData.values))
################################################################################################################################################
#display data
data_visualize(data, headernames)
check_feature(data)

data['Status'] = data.apply(convert_status_to_description, axis=1)
print (data.head(10))
data = data.drop(['Status'], axis=1)

data['BMI'] = data.apply(calulate_bmi, axis=1)
data = data[coln]

#For binary classification
#data['Risk'] = data.apply(calculate_risk, axis=1)
#data = data[coln_binary]

#data.to_csv(r'export_dataframe.csv', index = False, header=True)

data_visualize(data, headernames)

corr_sort = find_feature_reduced_matrix(data.corr())
print ("Strong dependency,", corr_sort)

to_drop = find_feature_reduced_matrix(data.corr())
print ("Remove dependency,", to_drop)
data.drop(to_drop, axis=1, inplace=True)

print (data.columns)

features = data.shape[1]-1
print (data.head(10))
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

#shuffle data
np.random.shuffle(selected_columns) 
print ("Data shape after shuffling:", selected_columns.shape)

################################################################################################################################################
#ML-Data Ready

fold = 5
X, y = selected_columns[:, :features], selected_columns[:,features]
#print (X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=33)

################################################################################################################################################
#ML-SVM
print ("SVM............linear........")
svm_linear_accuracy(X_train, X_test, y_train, y_test, fold)

################################################################################################################################################
#ML-SVM
#print ("SVM............rbf........")
#svm_nonlinear_accuracy(X_train, X_test, y_train, y_test, fold)

################################################################################################################################################
#ML-NB
#print ("NB..........GNB..........")
#NB_accuracy_gaussian(X_train, X_test, y_train, y_test, fold)

################################################################################################################################################
#ML-NB
#print ("NB..........BNB..........")
#NB_accuracy_barnoulli(X_train, X_test, y_train, y_test, fold)

################################################################################################################################################
#ML-NB
#print ("NB..........MNB..........")
#NB_accuracy_complement(X_train, X_test, y_train, y_test, fold)

################################################################################################################################################
#ML-KNN
#print ("KNN.......................")
#knn_accuracy(X_train, X_test, y_train, y_test, fold)

################################################################################################################################################
#ML-DT-gini
#print ("DT.......gini.............")
#DT_gini_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)

################################################################################################################################################
#ML-DT-entropy
#print ("DT.......entropy.............")
#DT_entropy_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)

################################################################################################################################################
#ML-RF
#print ("RF....................")
#RF_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold)

################################################################################################################################################
#ML-Ensemble-1
#print ("automatic_workflow........pipeline......LinearDiscriminantAnalysis......")
#automatic_workflow_lda(X_train, X_test, y_train, y_test, fold)

#####################################################multi-class classification###############################################################
#ML-compare multi-class classification
print ("Compare Machine Learning Algorithms Consistently....................")
print ("5 fold cross validation...")
#ensemble_all_general(X, y, fold = 5)
plot_ml_model(X, y, fold = 5) 

#print ("10 fold cross validation...")
#ensemble_all_general(X, y, fold = 10)
#plot_ml_model(X, y, fold = 10)

print("Grid search fold=5")
grid_search(X,y,fold=5)
#print("Grid search fold=10")
#grid_search(X,y,fold=10)
################################################################################################################################################
#ML-compare
#print ("Voting Ensemble Algorithms....................")
#ensemble_voting_classifier(X_train, X_test, y_train, y_test, fold)

################################################################################################################################################
#ML-Parameter tuning
#print ("Grid Search....................")
#grid_search(X_train, X_test, y_train, y_test, fold)

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

