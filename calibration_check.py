# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:22:10 2020

@author: ayanca
"""
# SVM reliability diagram
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from matplotlib import pyplot
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# fit a model
model = SVC()
model.fit(trainX, trainy)
# predict probabilities
probs = model.decision_function(testX)
# reliability diagram
fop, mpv = calibration_curve(testy, probs, n_bins=10, normalize=True)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
pyplot.plot(mpv, fop, marker='.')
pyplot.show()

# SVM reliability diagram - tuned
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# fit a model
model = SVC()
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(trainX, trainy)
# predict probabilities
probs = calibrated.predict_proba(testX)[:, 1]
# reliability diagram
fop, mpv = calibration_curve(testy, probs, n_bins=10, normalize=True)
#print (fop,mpv)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
pyplot.plot(mpv, fop, marker='.')
pyplot.show()

#####combined#############
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
 
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# uncalibrated predictions
yhat_uncalibrated = uncalibrated(trainX, testX, trainy)
print (yhat_uncalibrated.shape)
# calibrated predictions
yhat_calibrated = calibrated(trainX, testX, trainy)
yhat_calibrated2 = calibrated2(trainX, testX, trainy)
# reliability diagrams
fop_uncalibrated, mpv_uncalibrated = calibration_curve(testy, yhat_uncalibrated, n_bins=10, normalize=True)
fop_calibrated, mpv_calibrated = calibration_curve(testy, yhat_calibrated, n_bins=10)
fop_calibrated2, mpv_calibrated2 = calibration_curve(testy, yhat_calibrated2, n_bins=10)
# plot perfectly calibrated
fig = pyplot.figure(figsize = (14,8))
fig.suptitle('Calibration check of SVC')
pyplot.plot([0, 1], [0, 1], linestyle='--', color='black')
# plot model reliabilities
pyplot.plot(mpv_uncalibrated, fop_uncalibrated, marker='.', label = 'Uncalibrated')
pyplot.plot(mpv_calibrated, fop_calibrated, marker='.', label = 'Calibrated-sigmoid')
pyplot.plot(mpv_calibrated2, fop_calibrated2, marker='.', label = 'Calibrated-isotonic')
pyplot.legend()
pyplot.show()