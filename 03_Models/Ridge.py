# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os
import math
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
filenames.append("TimeSeriesCharac.csv")
#filenames.append("ARMAX.csv")

## loop through all files
for file in filenames:
    dataForRegression = pd.read_csv(file, parse_dates=['Time'], date_parser=dateparse)
    dataForRegression = dataForRegression.set_index('Time')
    if (file == "SnapZero.csv"):
        print("\n### These are the results of the SnapZero Ridge Regression ###")
    if (file == "SnapLag.csv"):
        print("\n### These are the results of the SnapLag Ridge Regression ###")
    if (file == "TimeSeriesCharac.csv"):
        print("\n### These are the results of the TimeSeriesCharac Ridge Regression ###")
    if (file == "ARMAX.csv"):
        print("\n### These are the results of the ARMAX Ridge Regression ###")

    #Aufteilen in Predictors und Targets
    dataForRegression_X = dataForRegression.iloc[:,:len(dataForRegression.columns)-1]
    dataForRegression_y = dataForRegression.loc[:,'Feinheit':]

    #Standardisieren der Trainingsdaten
    dataForRegression_X = pd.DataFrame(preprocessing.scale(dataForRegression_X))
    dataForRegression_X = dataForRegression_X.set_index(dataForRegression.iloc[:,:len(dataForRegression.index)].index)
    dataForRegression_X.columns = dataForRegression.iloc[:, :len(dataForRegression.columns) - 1].columns

    #Initialisierung der Error-Funktion
    scorer = make_scorer(score_func=errorFunction, greater_is_better=False)

    #Cross Validation von Ridge Parameters
    alphas = np.array([0, 1e-20, 1e-10, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 1000, 10000])
    alphas_grid = dict(alpha=alphas)
    clf_ridge = linear_model.Ridge()
    grid = GridSearchCV(estimator=clf_ridge, param_grid=alphas_grid, cv=5, scoring=scorer)
    grid.fit(dataForRegression_X, dataForRegression_y)
    print("\nRSME k-folded (k=5) Ridge-Regressions with different alpha:")
    print(*grid.grid_scores_, sep="\n")
    print("\nThe best alpha for Regression is:", grid.best_estimator_.alpha)

    #Aufteilen von trainingData in Subsets von Trainings- und "Test"-Trainingsdaten mit Parametern seed & test_size
    seed = 1
    test_size = 0.3
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataForRegression_X, dataForRegression_y, test_size=test_size, random_state=seed)

    #Defintion verschiedener Modelle
    ridge = linear_model.Ridge(alpha=grid.best_estimator_.alpha)

    #Auswahl des Modells
    clf1 = ridge

    #training of classifier
    clf1.fit(X_train, y_train)

    #Prediction des Subsets von "Test"-Trainingsdaten mit clf1
    prediction_clf1 = pd.DataFrame(clf1.predict(X_test))
    prediction_clf1 = prediction_clf1.set_index(X_test.index)
    prediction_clf1.columns = ['Predictions']
    prediction_clf1_solution = pd.concat([X_test, prediction_clf1, y_test], axis=1, join_axes=[X_test.index])
    print("Prediction using ridge (clf1): ")
    print(prediction_clf1_solution)

    ###Berechnung des Prediktion-Errors
    error_clf1 = clf1.score(X_train, y_train)
    print("R^2 of ridge (clf1) on training data: ", error_clf1)
    errorFunction_clf1 = errorFunction(prediction_clf1, y_test)
    print("Error-Function of ridge (clf1) on test data: ", errorFunction_clf1)
    errorUsingMedian = errorFunction([np.mean(y_test) for i in range(0,len(y_test))], y_test)
    print("Error-Function of always predicting mean: ", errorUsingMedian)

    #Prediction von allen Werten für Auswertung in csv
    predictionAll_clf1 = pd.DataFrame(clf1.predict(dataForRegression_X))
    predictionAll_clf1 = predictionAll_clf1.set_index(dataForRegression_X.index)
    predictionAll_clf1.columns = ['Predictions']
    predictionAll_clf1_solution = pd.concat([predictionAll_clf1, dataForRegression_y], axis=1, join_axes=[predictionAll_clf1.index])
    koeffizienten = pd.DataFrame(np.concatenate((np.array([dataForRegression_X.columns]),clf1.coef_), axis=0)).transpose()
    koeffizienten.columns = ['Name_Koeffizienten', 'Wert_Koeffizienten']
    if (file == "SnapZero.csv"):
        predictionAll_clf1_solution.to_csv("RidgeSnapZeroResults.csv")
        koeffizienten.to_csv("RidgeSnapZeroCoef.csv")
    if (file == "SnapLag.csv"):
        predictionAll_clf1_solution.to_csv("RidgeSnapLagResults.csv")
        koeffizienten.to_csv("RidgeSnapLagCoef.csv")
    if (file == "TimeSeriesCharac.csv"):
        predictionAll_clf1_solution.to_csv("RidgeTimeSeriesCharacResults.csv")
        koeffizienten.to_csv("RidgeTimeSeriesCharacCoef.csv")
    if (file == "ARMAX.csv"):
        predictionAll_clf1_solution.to_csv("RidgeARMAXResults.csv")
        koeffizienten.to_csv("RidgeARMAXCoef.csv")