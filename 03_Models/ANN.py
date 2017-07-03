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
#filenames.append("ARMAX.csv")


for i in filenames:


    ##### Set seed and test_size
    seed = 1
    test_size = 0.2


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


    ##### Rescale data (often referred to as normalization) http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/
    minmaxscaler = MinMaxScaler(feature_range=(0, 1))
    rescaledDataForRegression = minmaxscaler.fit_transform(dataForRegression)
    dataForRegression_X_minmaxscaled = rescaledDataForRegression[:,0:numberOfPredictors]
    dataForRegression_Y_minmaxscaled = rescaledDataForRegression[:,numberOfPredictors]
    X_train_minmaxscaled, X_test_minmaxscaled, Y_train_minmaxscaled, Y_test_minmaxscaled = cross_validation.train_test_split(dataForRegression_X_minmaxscaled, dataForRegression_Y_minmaxscaled, test_size=test_size, random_state=seed)


    ##### Standardize data (http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/)
    standardscaler = StandardScaler()
    standardizedDataForRegression = standardscaler.fit_transform(dataForRegression)
    dataForRegression_X_standardized = standardizedDataForRegression[:,0:numberOfPredictors]
    dataForRegression_Y_standardized = standardizedDataForRegression[:,numberOfPredictors]
    X_train_standardized, X_test_standardized, Y_train_standardized, Y_test_standardized = cross_validation.train_test_split(dataForRegression_X_standardized, dataForRegression_Y_standardized, test_size=test_size, random_state=seed)


    ##### Define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(numberOfPredictors, input_dim=numberOfPredictors, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        return model


    ##### Fit baseline model on training data and make predictions using test data
    #estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=400, batch_size=5, verbose=2)
    #estimator.fit(X_train, Y_train, batch_size=5, epochs=400, verbose=2)
    #predictions = estimator.predict(X_test)
    current_model = "ANN" + i[:-4] + "Results.csv"
    #print(current_model + " has RMSE of: " + str(errorFunction(predictions, Y_test)))


    ##### Fit baseline model on minmaxscaled training data, make prediction on test data and inverse transform it
    #estimator_minmaxscaled = KerasRegressor(build_fn=baseline_model, nb_epoch=180, batch_size=5, verbose=2)
    #estimator_minmaxscaled.fit(X_train_minmaxscaled, Y_train_minmaxscaled, batch_size=5, epochs=180, verbose=2)
    #predictions_minmaxscaled = estimator_minmaxscaled.predict(X_test_minmaxscaled)[:, None]
    #predictions_minmaxscaled_matrix = np.hstack((np.ones((len(X_test), numberOfPredictors)), predictions_minmaxscaled))
    #predictions_minmaxinversed = minmaxscaler.inverse_transform(predictions_minmaxscaled_matrix)[:,numberOfPredictors]
    #print(current_model + " (minmaxscaled) has RMSE of: " + str(errorFunction(predictions_minmaxinversed, Y_test)))


    ##### Fit baseline model on standardized training data, make prediction on test data and inverse transform it
    estimator_standardized = KerasRegressor(build_fn=baseline_model)
    estimator_standardized.fit(X_train_standardized, Y_train_standardized, batch_size=3, epochs=300, verbose=2)
    predictions_standardized = estimator_standardized.predict(X_test_standardized)[:, None]
    predictions_standardized_matrix = np.hstack((np.ones((len(X_test), numberOfPredictors)), predictions_standardized))
    predictions_standardizedinversed = standardscaler.inverse_transform(predictions_standardized_matrix)[:,numberOfPredictors]
    print(current_model + " (standardized) has RMSE of: " + str(errorFunction(predictions_standardizedinversed, Y_test)))


    ##### Save dataframe with results
    to_save = pd.DataFrame(
        {'Time': dFR_X_test.index,
         'Predictions': predictions_standardizedinversed,
         'Feinheit': Y_test})
    to_save = to_save.set_index('Time')
    #to_save.to_csv(current_model)


    ##### Just for fun: using all X data and 10fold cross validation, calculate RMSE
    #estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=2)
    #kfold = KFold(n_splits=10, random_state=seed) #Nacho --> kein k-Fold
    #results = cross_val_score(estimator,X ,Y , cv=kfold)
    #print(filename_to_save + " with entire X/Y dataset and 10fold cross validation has RMSE of: " + str(math.sqrt(results.mean())))















    # define the model for a deeper network
    #def deeper_model():
        # create model
        #modelD = Sequential()
        #modelD.add(Dense(numberOfPredictors, input_dim=numberOfPredictors, kernel_initializer='normal', activation='relu'))
        #modelD.add(Dense(6, kernel_initializer='normal', activation='relu'))
        #modelD.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        #modelD.compile(loss='mean_squared_error', optimizer='adam')
        #return modelD

    #np.random.seed(seed)
    #estimatorsD = []
    #estimatorsD.append(('standardize', StandardScaler()))
    #estimatorsD.append(('mlp', KerasRegressor(build_fn=deeper_model, epochs=50, batch_size=5, verbose=0)))
    #pipelineD = Pipeline(estimatorsD)
    #kfoldD = KFold(n_splits=10, random_state=seed)
    #resultsD = cross_val_score(pipeline, X, Y, cv=kfoldD)
    #print("Larger " + i + ": %.2f (%.2f) MSE" % (resultsD.mean(), resultsD.std()))

    # define wider model
    #def wider_model():
        # create model
        #modelW = Sequential()
        #modelW.add(Dense(20, input_dim=numberOfPredictors, kernel_initializer='normal', activation='relu'))
        #modelW.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        #modelW.compile(loss='mean_squared_error', optimizer='adam')
        #return modelW

    #np.random.seed(seed)
    #estimatorsW = []
    #estimatorsW.append(('standardize', StandardScaler()))
    #estimatorsW.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
    #pipelineW = Pipeline(estimatorsW)
    #kfoldW = KFold(n_splits=10, random_state=seed)
    #resultsW = cross_val_score(pipelineW, X, Y, cv=kfoldW)
    #print("Wider " + i + ": %.2f (%.2f) MSE" % (resultsW.mean(), resultsW.std()))