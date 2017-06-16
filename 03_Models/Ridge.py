# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os
from sklearn import cross_validation, linear_model
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import preprocessing
#from .Helper import errorFunction

#Defition einer Error-Funktion (RMSE)
def errorFunction(y,y_pred):
    accuracy = math.sqrt(mean_squared_error(y, y_pred))
    return accuracy

#Fetching training data set
felixOrLeo = "f"
if (felixOrLeo == "f"):
    pathData = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze"
    pathInterface = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Interface"
else:
    pathData = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"
    pathInterface = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Interface"

os.chdir(pathInterface)
cwd = os.getcwd()
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

## train the models for all files generated in 02_TrainingSetGeneration
filenames = []
filenames.append("SnapZero.csv")
filenames.append("SnapLag.csv")
#filenames.append("TimeSeriesCharac.csv")
#filenames.append("ARMAX.csv")

## loop through all files
for i in filenames:
    dataForRegression = pd.read_csv(i, parse_dates=['Time'], date_parser=dateparse)
    dataForRegression = dataForRegression.set_index('Time')


    #Aufteilen in Predictors und Targets
    dataForRegression_X = dataForRegression.ix[:,:len(dataForRegression.columns)-1]
    dataForRegression_y = dataForRegression.ix[:,'Feinheit':]

    #Standardisieren der Trainingsdaten
    dataForRegression_X = pd.DataFrame(preprocessing.scale(dataForRegression_X))
    dataForRegression_X = dataForRegression_X.set_index(dataForRegression.ix[:,:len(dataForRegression.index)].index)

    #Aufteilen von trainingData in Subsets von Trainings- und "Test"-Trainingsdaten mit Parametern seed & test_size
    seed = 1
    test_size = 0.3
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataForRegression_X, dataForRegression_y, test_size=test_size, random_state=seed)

    #Defintion verschiedener Modelle
    ridge1 = linear_model.Ridge(alpha=10)
    ridge2 = linear_model.Ridge(alpha=10)
    ridge3 = linear_model.Ridge(alpha=10)
    ridge4 = linear_model.Ridge(alpha=10)

    #Auswahl des Modells
    # if i 0 = 1:
    clf1 = ridge1

    #training of classifier
    clf1.fit(X_train, y_train)

    #Prediction des Subsets von "Test"-Trainingsdaten mit clf1
    prediction_clf1 = pd.DataFrame(clf1.predict(X_test))
    prediction_clf1 = prediction_clf1.set_index(X_test.index)
    prediction_clf1.columns = ['Predictions']
    prediction_clf1_solution = pd.concat([X_test, prediction_clf1, y_test], axis=1, join_axes=[X_test.index])
    print("Prediction using (clf1): ")
    print(prediction_clf1_solution)

    ###Berechnung des Prediktion-Errors
    error_clf1 = clf1.score(X_train, y_train)
    print("R^2 of chosen regression (clf1) on training data: ", error_clf1)
    errorFunction_clf1 = errorFunction(prediction_clf1, y_test)
    print("Error-Function of chosen regression (clf1) on test data: ", errorFunction_clf1)
    errorUsingMedian = errorFunction([np.mean(y_test) for i in range(0,len(y_test))], y_test)
    print("Error-Function of always predicting mean: ", errorUsingMedian)

    #Initialisierung der Error-Funktion
    scorer = make_scorer(score_func=errorFunction, greater_is_better=True)

    #Cross Validation von Ridge Parameters
    alphas = np.array([1e-5, 0.001, 0.1, 1, 10, 50, 100, 1000, 10000])
    alphas_grid = dict(alpha=alphas)
    clf_ridge = linear_model.Ridge()
    grid = GridSearchCV(estimator=clf_ridge, param_grid=alphas_grid, cv=5, scoring=scorer)
    grid.fit(dataForRegression_X, dataForRegression_y)
    print("\nRSME k-folded (k=5) Ridge-Regressions with different alpha:")
    print(*grid.grid_scores_, sep="\n")