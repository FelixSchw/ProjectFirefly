# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os
from sklearn import cross_validation, linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
import Helper as hlpr

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
filenames.append("ARX.csv")

## loop through all files
for file in filenames:
    dataForRegression = pd.read_csv(file, parse_dates=['Time'], date_parser=dateparse)
    dataForRegression = dataForRegression.set_index('Time')
    if (file == "SnapZero.csv"):
        print("\n### These are the results of the SnapZero Lasso Regression ###")
    if (file == "SnapLag.csv"):
        print("\n### These are the results of the SnapLag Lasso Regression ###")
    if (file == "TimeSeriesCharac.csv"):
        print("\n### These are the results of the TimeSeriesCharac Lasso Regression ###")
    if (file == "ARX.csv"):
        print("\n### These are the results of the ARMAX Lasso Regression ###")

    #Aufteilen in Predictors und Targets
    dataForRegression_X = dataForRegression.iloc[:,:len(dataForRegression.columns)-1]
    dataForRegression_y = dataForRegression.loc[:,'Feinheit':]

    #Standardisieren der Trainingsdaten
    dataForRegression_X = pd.DataFrame(preprocessing.scale(dataForRegression_X))
    dataForRegression_X = dataForRegression_X.set_index(dataForRegression.iloc[:,:len(dataForRegression.index)].index)
    dataForRegression_X.columns = dataForRegression.iloc[:, :len(dataForRegression.columns) - 1].columns

    #Initialisierung der Error-Funktion
    scorer = make_scorer(score_func=hlpr.errorFunction, greater_is_better=False)

    #Aufteilen von trainingData in Subsets von Trainings- und "Test"-Trainingsdaten mit Parametern seed & test_size
    seed = 1
    test_size = 0.2
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataForRegression_X, dataForRegression_y, test_size=test_size, random_state=seed)

    #Cross Validation von Ridge Parameters
    alphas = np.array([i*0.01 for i in range(0, 500, 1)])
    alphas_grid = dict(alpha=alphas)
    clf_lasso = linear_model.Lasso(tol=10)
    grid = GridSearchCV(estimator=clf_lasso, param_grid=alphas_grid, cv=5, scoring=scorer)
    grid.fit(X_train, y_train)
    print("\nRSME k-folded (k=5) Lasso-Regressions with different alpha:")
    print(*grid.grid_scores_, sep="\n")
    print("\nThe best alpha for Regression is:", grid.best_estimator_.alpha)

    #Plotten von Error-Function vs. Parameter
    import matplotlib.pyplot as plt
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.multiply(scores, (-1))
    plt.plot(alphas, scores)
    plt.show()

    # Plotten von Koeffizienten vs. Strafterm
    for alpha in alphas:
        if (alpha == 0):
            coefficients = np.array([]).reshape(0, len(dataForRegression_X.columns))
        ridge = linear_model.Ridge(alpha=alpha)
        ridge.fit(dataForRegression_X, dataForRegression_y)
        coefficients = np.append(coefficients,ridge.coef_, axis=0)

    import matplotlib.pyplot as plt
    plt.plot(alphas, coefficients[:, 0])
    plt.plot(alphas, coefficients[:, 1])
    plt.plot(alphas, coefficients[:, 2])
    plt.show()

    #Defintion verschiedener Modelle
    lasso = linear_model.Lasso(alpha=grid.best_estimator_.alpha)

    #Auswahl des Modells
    clf1 = lasso

    #training of classifier
    clf1.fit(X_train, y_train)

    #Prediction des Subsets von "Test"-Trainingsdaten mit clf1
    prediction_clf1 = pd.DataFrame(clf1.predict(X_test))
    prediction_clf1 = prediction_clf1.set_index(X_test.index)
    prediction_clf1.columns = ['Predictions']
    prediction_clf1_solution = pd.concat([X_test, prediction_clf1, y_test], axis=1, join_axes=[X_test.index])
    print("Prediction using lasso (clf1): ")
    #print(prediction_clf1_solution)

    ###Berechnung des Prediktion-Errors
    error_clf1 = clf1.score(X_train, y_train)
    print("R^2 of lasso (clf1) on training data: ", error_clf1)
    errorFunction_clf1 = hlpr.errorFunction(prediction_clf1, y_test)
    print("Error-Function of lasso (clf1) on test data: ", errorFunction_clf1)
    errorUsingMedian = hlpr.errorFunction([np.mean(y_test) for i in range(0,len(y_test))], y_test)
    print("Error-Function of always predicting mean: ", errorUsingMedian)

    # Predictions von Test-Set in .csv schreiben
    prediction_clf1_solution = prediction_clf1_solution[['Feinheit', 'Predictions']]
    koeffizienten = pd.DataFrame(np.concatenate((np.array([dataForRegression_X.columns]),np.array([clf1.coef_])), axis=0)).transpose()
    koeffizienten.columns = ['Name_Koeffizienten', 'Wert_Koeffizienten']
    if (file == "SnapZero.csv"):
        prediction_clf1_solution.to_csv("LassoSnapZeroResults.csv")
        koeffizienten.to_csv("LassoSnapZeroCoef.csv")
    if (file == "SnapLag.csv"):
        prediction_clf1_solution.to_csv("LassoSnapLagResults.csv")
        koeffizienten.to_csv("LassoSnapLagCoef.csv")
    if (file == "TimeSeriesCharac.csv"):
        prediction_clf1_solution.to_csv("LassoTimeSeriesCharacResults.csv")
        koeffizienten.to_csv("LassoTimeSeriesCharacCoef.csv")
    if (file == "ARX.csv"):
        prediction_clf1_solution.to_csv("LassoARXResults.csv")
        koeffizienten.to_csv("LassoARXCoef.csv")
