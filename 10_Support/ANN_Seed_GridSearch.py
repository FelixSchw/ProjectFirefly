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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

##### Defition einer Error-Funktion (RMSE)
def errorFunction(y, y_pred):
    accuracy = math.sqrt(mean_squared_error(y, y_pred))
    return accuracy


##### Set seed
np.random.seed(1)


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


    ##### Grid search http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    print("Grid_Search starts now at " + pd.datetime.now().ctime())
    grid_seeds = [i for i in range(0,1000)]
    grid_epochs = 310
    grid_batch = 1
    grid_neurons = 3
    grid_activation = 'tanh'
    grid_results = []
    grid_results.append("Seed,RMSE")

    for i in range(0, len(grid_seeds)):
        ##### Zwischenspeichern
        np.savetxt("Grid_Search_Results_Seed.txt", grid_results, delimiter=" ", fmt="%s")
        ##### Derive name
        grid_current_name = str(grid_seeds[i]) + ","
        ##### Define base model
        def baseline_model():
            model = Sequential()
            model.add(Dense(grid_neurons, input_dim=numberOfPredictors, kernel_initializer='normal',activation=grid_activation))
            model.add(Dense(1, kernel_initializer='normal'))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
            return model
        ##### Fit baseline model on standardized training data, make prediction on test data and inverse transform it --> yields 0.39
        np.random.seed(grid_seeds[i])
        estimator_standardized = KerasRegressor(build_fn=baseline_model)
        estimator_standardized.fit(X_train_standardized, Y_train_standardized, batch_size=grid_batch, epochs=grid_epochs, verbose=0)
        predictions_standardized = estimator_standardized.predict(X_test_standardized)[:, None]
        predictions_standardized_matrix = np.hstack((np.ones((len(X_test), numberOfPredictors)), predictions_standardized))
        predictions_standardizedinversed = standardscaler.inverse_transform(predictions_standardized_matrix)[:, numberOfPredictors]
        ##### Update grid_current_name, print it and append it to grid_results
        grid_current_name = grid_current_name + str(errorFunction(predictions_standardizedinversed, Y_test))
        print(grid_current_name)
        grid_results.append(grid_current_name)

    np.savetxt("Grid_Search_Results_Seed.txt", grid_results, delimiter=" ", fmt="%s")
    print(str(grid_results))
    print("Grid_Search finished now at " + pd.datetime.now().ctime())


