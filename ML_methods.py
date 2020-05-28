# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:33:14 2020

@author: ayanca
"""

from matplotlib import pyplot
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.impute  import SimpleImputer
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
#import pydotplus
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score, log_loss)
from sklearn.linear_model import LogisticRegression

def plot_Correlation(data, names):
    #'pearson', 'spearman', 'kendall'
    sns.set_style('whitegrid')
    correlations = data.corr()
    
    print (correlations)
    fig = pyplot.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    sns.heatmap(correlations, annot = True, cmap='coolwarm',linewidths=.1)
    pyplot.show()    
    
def remove_missing(data):
    print (data.isnull().any())
    missing_values = ["n/a", "na", "--","?", " ","NA"]
    data = data.replace(missing_values, np.nan)
    feat_miss = data.columns[data.isnull().any()]
    if feat_miss.size == 0:
        print ("Data is clean")
    else:
        print ("Missing data shape before:", feat_miss.shape)
        imputer = SimpleImputer(copy=True, fill_value=None, missing_values=np.nan, strategy='calculate_iris', verbose=0)
        data[feat_miss] = imputer.fit_transform(data[feat_miss])
        feat_miss = data.columns[data.isnull().any()]
        print ("Missing data shape after:", feat_miss.shape)
    return data
        
def data_visualize(data, headernames):
    pyplot.close('all')
    set_option('display.width', 100)
    set_option('precision', 2)

    print ("Original data shape:", data.shape) #dimension
    #print(data.head(50))
    print("Data Information: ", data.info())
    print("Describe data: ", data.describe()) #statistical summary of the data

    count_class = data.groupby('Index').size() #Index distribution
    print("count_class", count_class)

    count_class = data.groupby('Gender').size() #Gender distribution
    print("count_class_gender", count_class)

    plot_Correlation(data,headernames)
    #print ("Correlation:", correlations)
    print ("Skewness:", data.skew())   

    #set frame
    fig = pyplot.figure(figsize = (14,8))
    ax = fig.gca()

    #histogram
    data.hist(ax = ax)
    pyplot.show()
    #pyplot.savefig('hist.png')
    #density
    data.plot(kind='density', figsize= (14, 8), subplots=True, layout=(3,3), sharex=False)
    pyplot.show()
    #boxplot
    data.plot(kind = 'box', figsize=(14, 8), subplots = True, layout = (3,3), sharex = False, sharey = False)
    pyplot.show()
    #scatter
    scatter_matrix(data, alpha=0.2, figsize=(14, 8), diagonal='kde')
    pyplot.show()
    
def check_feature(df):
    male =len(df[df['Gender'] == 1])
    female = len(df[df['Gender']== 0])

    pyplot.figure(figsize=(14,8))

    # Data to plot
    labels = 'Male','Female'
    sizes = [male,female]
    colors = ['skyblue', 'yellowgreen']
    explode = (0, 0)  # explode 1st slice
 
    # Plot
    pyplot.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90) 
    pyplot.axis('equal')
    pyplot.show()
    
    # Data to plot
    pyplot.figure(figsize=(14,8))
    labels = 'Extremely Weak:0','Weak:1','Normal:2','Overweight:3', 'Obesity:4', 'Extreme Obesity:5'
    sizes = [len(df[df['Index'] == 0]),len(df[df['Index'] == 1]), len(df[df['Index'] == 2]),len(df[df['Index'] == 3]), len(df[df['Index'] == 4]), len(df[df['Index'] == 5])]
    colors = ['skyblue', 'yellowgreen','orange','gold', 'red', 'blue']
    explode = (0, 0,0,0,0,0)  # explode 1st slice
 
    # Plot
    pyplot.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=180) 
    pyplot.axis('equal')
    pyplot.show()
    
def find_high_dependency(correlation_matrix):
    # map features to their absolute correlation values
    corr = correlation_matrix.abs()
    # set equality (self correlation) as zero
    corr[corr == 1] = 0
    # of each feature, find the max correlation
    # and sort the resulting array in ascending order
    corr_cols = corr.max().sort_values(ascending=False)
    # display the highly correlated features
    return (corr_cols[corr_cols >= 0.9])

def find_feature_reduced_matrix(correlation_matrix):
    corr_matrix = correlation_matrix.abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] >= 0.9)]
    # Drop features 
    #df.drop(to_drop, axis=1, inplace=True)
    return to_drop
    
def convert_status_to_description(df):
    if df['Index'] == 0:
        return 'Extremely Weak'
    elif df['Index'] == 1:
        return 'Weak'
    elif df['Index'] == 2:
        return 'Normal'
    elif df['Index'] == 3:
        return 'Overweight'
    elif df['Index']== 4:
        return 'Obesity'
    elif df['Index'] == 5:
        return 'Extreme Obesity'  
    
def calculate_risk(df):
    if df['Index'] == 0:
        return 0
    elif df['Index'] == 1:
        return 0
    elif df['Index'] == 2:
        return 0
    elif df['Index'] == 3:
        return 1
    elif df['Index']== 4:
        return 1
    elif df['Index'] == 5:
        return 1
    
def calulate_bmi(df):
    return (df['Weight'] * df['Weight'])/df['Height']

def svm_linear_accuracy(X_train, X_test, y_train, y_test, fold):
    svc = SVC(kernel='linear', gamma ='auto', C=1.0)
    classifier = svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    #print ("mean_squared_error: ", mean_squared_error(y_test, y_pred))
    
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=['Gender', 'Height', 'Weight', 'Index'],
                                 cmap=pyplot.cm.Blues,
                                 normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)  
    pyplot.show()
    
    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))
    
    #calibration(X_train, y_train, X_test, y_test)
    
    results = cross_val_score(svc, X_train, y_train, cv = fold)
    print("After 5-fold: ", results.mean()*100)    
    #print (y_test.shape, y_pred.shape)
        
def svm_nonlinear_accuracy(X_train, X_test, y_train, y_test, fold):
    svc = SVC(kernel='rbf', gamma ='auto', C=1.0)
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(svc, X_train, y_train, cv = fold)
    print("After 5-fold: ", results.mean()*100)  
    
def NB_accuracy_gaussian(X_train, X_test, y_train, y_test, fold):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print ("mean_squared_error: ", mean_squared_error(y_test, y_pred))
    
    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(gnb, X_train, y_train, cv = fold)
    print("After 5-fold: ", results.mean()*100)

      
def NB_accuracy_barnoulli(X_train, X_test, y_train, y_test, fold):
    gnb = BernoulliNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(gnb, X_train, y_train, cv = fold)
    print("After 5-fold: ", results.mean()*100)
    
def NB_accuracy_complement(X_train, X_test, y_train, y_test, fold):
    gnb = ComplementNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(gnb, X_train, y_train, cv = fold)
    print("After 5-fold: ", results.mean()*100)
    
def DT_gini_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(clf, X_train, y_train, cv = fold)
    print("After 5-fold: ", results.mean()*100)

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4','5'])

    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #graph.write_png('Obesity_Tree_Gini.png')
    #Image(graph.create_png())
    
def DT_entropy_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold):
    #clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(clf, X_train, y_train, cv = fold)
    print("After 5-fold: ", results.mean()*100)

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4','5'])
    '''
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('Obesity_Tree_Entropy.png')
    Image(graph.create_png())
    '''
    
def RF_accuracy(X_train, X_test, y_train, y_test, feature_cols, fold):
    classifier = RandomForestClassifier(n_estimators = 50)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(classifier, X_train, y_train, cv = fold)

    print("After 5-fold: ", results.mean()*100)

    
def knn_accuracy(X_train, X_test, y_train, y_test, fold):
    classifier = KNeighborsClassifier(n_neighbors = 6)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print ('R Squared =',r2_score(y_test, y_pred))
    print ('MAE =', mean_absolute_error(y_test, y_pred))
    print ('MSE =',mean_squared_error(y_test, y_pred))

    results = cross_val_score(classifier, X_train, y_train, cv = fold)

    print("After 5-fold: ", results.mean()*100)

    
def automatic_workflow_lda(X_train, X_test, y_train, y_test, fold):
    #base estimators
    estimators = []
    
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('lda', LinearDiscriminantAnalysis()))
    
    model = Pipeline(estimators) 
    
    #kfold = KFold(n_splits = 20, random_state = 7)
    results = cross_val_score(model, X_train, y_train, cv = fold)
    print(results.mean()*100)
    
def ensemble_all_general(X, y, fold):
    models = []
    num_trees = 150
    seed = 7   
    
    est1 = SVC(kernel='linear', gamma ='auto', C=1.0)
    est2 = SVC(kernel='rbf', gamma ='auto', C=1.0)
    est3 = GaussianNB()
    est4 = BernoulliNB()
    est5 = ComplementNB()
    est6 = DecisionTreeClassifier()
    est7 = DecisionTreeClassifier(criterion="entropy")
    est8 = RandomForestClassifier(n_estimators = 50)
    est9 = KNeighborsClassifier(n_neighbors = 6)  
    est10 = BaggingClassifier(base_estimator = est6, n_estimators = num_trees, random_state = seed)
    est11 = AdaBoostClassifier(n_estimators = 50, random_state = seed)
    est12 = GradientBoostingClassifier(n_estimators = 150, random_state = seed)
    
    models.append(('SVM-1', est1))
    models.append(('SVM-2', est2))   
    models.append(('NB-1', est3))
    models.append(('NB-2', est4))
    models.append(('NB-3', est5))
    models.append(('DT-1', est6))
    models.append(('DT-2', est7))
    #models.append(('RF-1', est8))
    models.append(('RF-2', RandomForestClassifier()))
    #models.append(('KNN-1', est9))
    models.append(('KNN-2', KNeighborsClassifier()))    
    #models.append(('LDA', LinearDiscriminantAnalysis())) 
    #models.append(('bagging', est10))  
    #models.append(('adaboost', est11))     
    #models.append(('gradboost', est12)) 
    
    #plot_ml_model(models)
    
    # evaluate each model in turn
    #seed = 7
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        #ld = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results  = model_selection.cross_val_score(model, X, y, cv=fold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg) 
        
    # boxplot algorithm comparison
    fig = pyplot.figure(figsize= (16, 16))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()    

def plot_ml_model(X, y, fold):
    pyplot.close('all')   
    algos = ["SVM-linear","SVM-Kernel","GaussianNB","ComplementNB","DTree-gini","DTree-entropy","RF-50","RF-100","KNN-2", "KNN-6"]
       
    clfs = [SVC(kernel='linear'),
            SVC(kernel='rbf'),
            GaussianNB(),
            #BernoulliNB(),
            ComplementNB(),
            DecisionTreeClassifier(),
            DecisionTreeClassifier(criterion="entropy"),
            RandomForestClassifier(n_estimators = 50),
            RandomForestClassifier(n_estimators = 100),
            #RandomForestClassifier(n_estimators = 150),
            KNeighborsClassifier(n_neighbors = 2),  
            KNeighborsClassifier(n_neighbors = 6)]
    
    cv_results = []
    
    scoring = 'accuracy'
    for classifiers in clfs:
        cv_score = cross_val_score(classifiers,X,y,cv=fold,scoring=scoring)
        cv_results.append(cv_score.mean())
        
    cv_mean = pd.DataFrame(cv_results,index=algos)
    cv_mean.columns=["Accuracy"]
    print (cv_mean.sort_values(by="Accuracy",ascending=False))
    cv_mean.plot.bar(figsize=(10,5))
    
    #scatter plot
    scores=cv_mean["Accuracy"]
    #create traces
    trace1 = go.Scatter(x = algos, y= scores, name='Algortms Name', marker =dict(color='rgba(0,255,0,0.5)',
               line =dict(color='rgb(0,0,0)',width=2)),
                text=algos
                )
    data = [trace1]

    layout = go.Layout(barmode = "group", xaxis= dict(title= 'ML Algorithms',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Prediction Scores',ticklen= 5,zeroline= False))
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    pyplot.show()   

def grid_search(X, y, fold):
    #models = []    
    alphas = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': alphas, 'gamma' : gammas}  
    
    '''
    est1 = SVC(kernel='linear', gamma ='auto', C=1.0)
    est2 = SVC(kernel='rbf', gamma ='auto', C=1.0)
    est3 = GaussianNB()
    est4 = BernoulliNB()
    est5 = ComplementNB()
    est6 = DecisionTreeClassifier()
    est7 = DecisionTreeClassifier(criterion="entropy")
    est8 = RandomForestClassifier(n_estimators = 50)
    est9 = KNeighborsClassifier(n_neighbors = 6)    
    
    models.append(('SVM-1', est1))
    #models.append(('SVM-2', est2))   
    models.append(('NB-1', est3))
    #models.append(('NB-2', est4))
    #models.append(('NB-3', est5))
    models.append(('DT-1', est6))
    models.append(('DT-2', est7))
    models.append(('RF-1', est8))
    #models.append(('RF-2', RandomForestClassifier()))
    models.append(('KNN-1', est9))
    models.append(('KNN-2', KNeighborsClassifier()))    
    models.append(('LDA', LinearDiscriminantAnalysis()))  
    models.append(('SVM-linear', est1))
    
    for name, model in models:        
        grid = GridSearchCV(estimator = model, param_grid = param_grid)
        grid.fit(X, y)       
        msg = "%s: %f (%f)" % (name, grid.best_score_, grid.best_estimator_.alpha)
        print(msg)
    '''
    scoring = 'accuracy'    
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=fold, scoring = scoring)
    grid_search.fit(X, y)
    #msg = "%f (%f)" % (grid_search.best_score_, grid_search.best_estimator_.alpha)
    print(grid_search.best_params_)
    print("The best parameters are %s with a score of %0.2f" % (grid_search.best_params_, grid_search.best_score_))
    
def ensemble_voting_classifier(X_train, X_test, y_train, y_test, fold):   
    models = []
    num_trees = 150
    seed = 7
    
    est1 = SVC(kernel='linear', gamma ='auto', C=1.0)
    est2 = SVC(kernel='rbf', gamma ='auto', C=1.0)
    est3 = GaussianNB()
    #est4 = BernoulliNB()
    #est5 = ComplementNB()
    est6 = DecisionTreeClassifier()
    #est7 = DecisionTreeClassifier(criterion="entropy")
    est8 = RandomForestClassifier() #RandomForestClassifier(n_estimators = num_trees, max_features = max_features)
    est9 = KNeighborsClassifier()  
    #est10 = BaggingClassifier(base_estimator = est6, n_estimators = num_trees, random_state = seed)
    #est11 = AdaBoostClassifier(n_estimators = 50, random_state = seed)
    #est12 = GradientBoostingClassifier(n_estimators = 150, random_state = seed)
    
    #models.append(('SVM-1', est1))
    #models.append(('SVM-2', est2))   
    #models.append(('NB-1', est3))
    #models.append(('NB-2', est4))
    #models.append(('NB-3', est5))
    models.append(('DT-1', est6))
    #models.append(('DT-2', est7))
    #models.append(('RF-1', est8))
    #models.append(('RF-2', RandomForestClassifier()))
    #models.append(('KNN-1', est9))
    #models.append(('KNN-2', KNeighborsClassifier()))    
    #models.append(('LDA', LinearDiscriminantAnalysis()))  
    #models.append(('bagging', est10))  
    #models.append(('adaboost', est11)) 
    
    #evaluate each model in turn
    #ensemble = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 150, random_state = 7)
    #ensemble = VotingClassifier(models)
    #ensemble = ExtraTreesClassifier(n_estimators = 150, random_state = 7)
    #ensemble = RandomForestClassifier(n_estimators = 50, random_state = 7)
    #ensemble = AdaBoostClassifier(n_estimators = 200, random_state = 7)
    ensemble = GradientBoostingClassifier(n_estimators = 200, random_state = 7)
    results = cross_val_score(ensemble, X_train, y_train, cv = fold)
    #print ("starts")
    print(results.mean()*100, results.std())
    #print ("ends")
   
# predict uncalibrated probabilities
def uncalibrated(trainX, testX, trainy):
	# fit a model
	model = SVC()
	model.fit(trainX, trainy)
	# predict probabilities
	return model.decision_function(testX)

# predict calibrated probabilities
def calibrated(trainX, testX, trainy):
	# define model
	model = SVC()
	# define and fit calibration model
	calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5) #method='isotonic' or 'sigmoid'
	calibrated.fit(trainX, trainy)
	# predict probabilities
	return calibrated.predict_proba(testX)[:, 1]

# predict calibrated probabilities
def calibrated2(trainX, testX, trainy):
	# define model
	model = SVC()
	# define and fit calibration model
	calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5) #method='isotonic' or 'sigmoid'
	calibrated.fit(trainX, trainy)
	# predict probabilities
	return calibrated.predict_proba(testX)[:, 1]

def calibration_check(trainX, trainy, testX, testy):
    # uncalibrated predictions
    yhat_uncalibrated = uncalibrated(trainX, testX, trainy)
    print (yhat_uncalibrated.shape)
    # calibrated predictions
    yhat_calibrated = calibrated(trainX, testX, trainy)    
    yhat_calibrated2 = calibrated2(trainX, testX, trainy)
    # reliability diagrams
    fop_uncalibrated, mpv_uncalibrated = calibration_curve(testy, yhat_uncalibrated, n_bins=5, normalize=True)
    fop_calibrated, mpv_calibrated = calibration_curve(testy, yhat_calibrated, n_bins=5)
    fop_calibrated2, mpv_calibrated2 = calibration_curve(testy, yhat_calibrated2, n_bins=5)
    # plot perfectly calibrated
    fig = pyplot.figure(figsize = (14,8))
    fig.suptitle('Calibration check')
    pyplot.plot([0, 1], [0, 1], linestyle='--', color='black')
    # plot model reliabilities
    pyplot.plot(mpv_uncalibrated, fop_uncalibrated, marker='.', label = 'Uncalibrated')
    pyplot.plot(mpv_calibrated, fop_calibrated, marker='.', label = 'Calibrated-sigmoid')
    pyplot.plot(mpv_calibrated2, fop_calibrated2, marker='.', label = 'Calibrated-isotonic')
    pyplot.legend()
    pyplot.show()
    
def calibration(X_train, y_train, X_test, y_test, est, name, fig_index): 
    max = y_test.max()
    if (y_train.max() > y_test.max()):
        max = y_train.max()
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=5, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=5, method='sigmoid')

    # Logistic regression with no calibration as baseline
    #lr = LogisticRegression(C=1., solver='lbfgs')

    fig = pyplot.figure(fig_index, figsize=(9, 9))
    ax1 = pyplot.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = pyplot.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=max)
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    pyplot.tight_layout()
    
def calibration_model_compare(X_train, y_train, X_test, y_test):
    max = y_test.max()
    if (y_train.max() > y_test.max()):
        max = y_train.max()
       
    svc_linear = SVC(kernel='linear')
    svc_non_linear = SVC(kernel='rbf')
    gnb = GaussianNB() 
    lr = LogisticRegression() #C=1., solver='lbfgs'
    dtree_gini = DecisionTreeClassifier()
    dtree_entropy = DecisionTreeClassifier(criterion="entropy")
    rf_50 = RandomForestClassifier(n_estimators = 50)
    rf_100 = RandomForestClassifier(n_estimators = 100)
    knn_2 = KNeighborsClassifier(n_neighbors = 2)
    knn_6 = KNeighborsClassifier(n_neighbors = 6)
 
    pyplot.figure(figsize=(9, 9))
    ax1 = pyplot.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = pyplot.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(svc_linear, 'svc_linear'),
                      (svc_non_linear, 'svc_non_linear'),
                      (gnb, 'Naive Bayes'),
                      (lr, 'Logistic Regression'),
                      (dtree_gini, 'dtree_gini'),
                      (dtree_entropy, 'dtree_entropy'),
                      (rf_50, 'Random Forest-50'),
                      (rf_100, 'Random Forest-100'),
                      (knn_2, 'KNN_2'),
                      (knn_6, 'KNN_6')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=max)
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
    
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name, ))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    pyplot.tight_layout()