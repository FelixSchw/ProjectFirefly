# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import math
import array
import os
#from scipy.stats import skew
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.constraints import maxnorm

##### Defition einer Error-Funktion (RMSE)
def errorFunction(y, y_pred):
    accuracy = math.sqrt(mean_squared_error(y, y_pred))
    return accuracy



##### Fetching training data set
felixOrLeo = "l"
if (felixOrLeo == "f"):
    pathData = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze"
    pathInterface = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Interface"
else:
    pathData = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"
    pathInterface = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Interface"
os.chdir(pathInterface)
cwd = os.getcwd()
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


##### train the models for all files generated in 02_TrainingSetGeneration
filenames = []
filenames.append("SnapZero.csv")
#filenames.append("SnapLag.csv")
#filenames.append("TimeSeriesCharac.csv")
#filenames.append("ARX.csv")


for i in filenames:


    ##### Set seed, test_size & current model name
    current_model = "ANN" + i[:-4] + "Results.csv"
    test_size = 0.2
    seed = 1


    ##### Read file with NN data, set index, and calculate # of predictors
    dataForRegression = pd.read_csv(i, parse_dates=['Time'], date_parser=dateparse)
    dataForRegression = dataForRegression.set_index('Time')
    numberOfPredictors = (len(dataForRegression.columns) - 1)


    ##### Split NN data into test and training data
    dataForRegression_X = dataForRegression.iloc[:,0:numberOfPredictors]
    dataForRegression_Y = dataForRegression.iloc[:,numberOfPredictors]
    X = dataForRegression_X.values ### only for k-fold cross validation
    Y = dataForRegression_Y.values ### only for k-fold cross validation
    dFR_X_train, dFR_X_test, dFR_Y_train, dFR_Y_test = cross_validation.train_test_split(dataForRegression_X, dataForRegression_Y, test_size=test_size, random_state=seed)
    X_train, X_test, Y_train, Y_test = dFR_X_train.values, dFR_X_test.values, dFR_Y_train.values, dFR_Y_test.values


    ##### Standardize data (http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/)
    standardscaler = StandardScaler()
    standardizedDataForRegression = standardscaler.fit_transform(dataForRegression)
    dataForRegression_X_standardized = standardizedDataForRegression[:,0:numberOfPredictors]
    dataForRegression_Y_standardized = standardizedDataForRegression[:,numberOfPredictors]
    X_train_standardized, X_test_standardized, Y_train_standardized, Y_test_standardized = cross_validation.train_test_split(dataForRegression_X_standardized, dataForRegression_Y_standardized, test_size=test_size, random_state=seed)


    ##### Define base model
    def baseline_model(neurons=1, dropout_rate=0.0, weight_constraint=0, activation='relu'):
        model = Sequential()
        model.add(Dense(neurons, input_dim=numberOfPredictors, kernel_initializer='normal', activation=activation, kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model


    ##### Set seed for reproducability
    np.random.seed(1)


    ##### Grid search http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    model = KerasClassifier(build_fn=baseline_model, verbose=0)


    ##### Define the grid search parameters
    batch_size = [5, 10, 50, 100]
    epochs = [10, 50, 100, 400]
    neurons = [1, 3, 5, 10, 20]
    activation = ['relu', 'tanh', 'softmax']
    weight_constraint = [1, 3, 5]
    dropout_rate = [0.0, 0.3, 0.6, 0.9]


    ##### Create file for output
    grid_results = []


    ##### Fit with parameters
    param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, activation=activation, weight_constraint=weight_constraint, dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, Y_train)


    ##### Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        grid_results.append("%f (%f) with: %r" % (mean, stdev, param))


    ##### Save results in file
    np.savetxt("Mystery_GridSearchResults.txt", grid_results, delimiter=" ", fmt="%s")