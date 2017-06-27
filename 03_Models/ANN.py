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
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

#Fetching training data set
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

## train the models for all files generated in 02_TrainingSetGeneration
filenames = []
filenames.append("SnapZero.csv")
filenames.append("SnapLag.csv")
filenames.append("TimeSeriesCharac.csv")
#filenames.append("ARMAX.csv")

## loop through all files
for i in filenames:
    dataForRegression = pd.read_csv(i, parse_dates=['Time'], date_parser=dateparse)
    dataForRegression = dataForRegression.set_index('Time')

    #Vorgehen Tutorial
    # load dataset
    #dataframe = pd.read_csv("/Users/leopoldspenner/Documents/python/housing.csv", delim_whitespace=True, header=None)
    #datasett = dataframe.values
    # split into input (X) and output (Y) variables
    #XX = datasett[:,0:13]
    #YY = datasett[:,13]

    #Standardisieren der Trainingsdaten
    #dataForRegression_X = pd.DataFrame(preprocessing.scale(dataForRegression_X))
    #dataForRegression_X = dataForRegression_X.set_index(dataForRegression.ix[:,:len(trainingData.columns)].index)


    dataset = dataForRegression.values
    numberOfPredictors = (len(dataForRegression.columns)-1)
    X = dataset[:,0:numberOfPredictors]
    Y = dataset[:,numberOfPredictors]
    seed = 7

    print("Data prep of " + i + " completet. Starting NN training")

    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(numberOfPredictors, input_dim=numberOfPredictors, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    # evaluate model with original dataset
    estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator,X ,Y , cv=kfold)
    print("Original " + i + ": %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # fit model with entire X as training set and make predictions
    estimator.fit(X,Y)
    predictions = estimator.predict(X)
    to_save = pd.DataFrame(
        {'Time': dataForRegression.index,
         'Predictions': predictions,
         'Feinheit': Y
         }
    )
    to_save = to_save.set_index('Time')
    filename_to_save = "ANN" + i[:-4] + "Results.csv"
    to_save.to_csv(filename_to_save)

    # standardize dataset
    #std_scale = preprocessing.StandardScaler().fit(X)
    #X_std = std_scale.transform(X)
    #http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#standardization-and-min-max-scaling

    # evaluate model with standardized dataset
    #estimatorS = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
    #kfold = KFold(n_splits=10, random_state=seed)
    #resultsS = cross_val_score(estimatorS, X_std, Y, cv=kfold)
    #print("Standardized " + i + ": %.2f (%.2f) MSE" % (resultsS.mean(), resultsS.std()))

    # evaluate model with standardized dataset
    #np.random.seed(seed)
    #estimatorsS = []
    #estimatorsS.append(('standardize', StandardScaler()))
    #estimatorsS.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    #pipelineS = Pipeline(estimatorsS)
    #kfoldS = KFold(n_splits=10, random_state=seed)
    #resultsS = cross_val_score(pipelineS, X, Y, cv=kfoldS)
    #print("Standardized " + i + ": %.2f (%.2f) MSE" % (resultsS.mean(), resultsS.std()))

    # define the model for a deeper network
    #def larger_model():
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
    #estimatorsD.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
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