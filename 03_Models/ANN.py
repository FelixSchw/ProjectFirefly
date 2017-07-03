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
#filenames.append("ARMAX.csv")


for i in filenames:


    ##### Read file with NN data, set index, and calculate # of predictors
    dataForRegression = pd.read_csv(i, parse_dates=['Time'], date_parser=dateparse)
    dataForRegression = dataForRegression.set_index('Time')
    numberOfPredictors = (len(dataForRegression.columns) - 1)


    ##### Split NN data into input (X) and output(Y) data
    dataForRegression_X = dataForRegression.iloc[:,0:numberOfPredictors]
    dataForRegression_Y = dataForRegression.iloc[:,numberOfPredictors]
    X = dataForRegression_X.values
    Y = dataForRegression_Y.values


    ##### Set seed and test_size
    seed = 1
    test_size = 0.2


    ##### Split dataForRegression in subsets of test & training data with parameters seed & test-size
    dFR_X_train, dFR_X_test, dFR_Y_train, dFR_Y_test = cross_validation.train_test_split(dataForRegression_X, dataForRegression_Y, test_size=test_size, random_state=seed)
    X_train, X_test, Y_train, Y_test = dFR_X_train.values, dFR_X_test.values, dFR_Y_train.values, dFR_Y_test.values
    print("Now starting: Training with X_train.shape=" + str(X_train.shape) + " and predicting with X_test.shape=" + str(X_test.shape))


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
    estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=2)
    estimator.fit(X_train, Y_train, batch_size=5, epochs=200, verbose=2)
    predictions = estimator.predict(X_test)
    current_model = "ANN" + i[:-4] + "Results.csv"
    print(current_model + " has RMSE of: " + str(errorFunction(predictions, Y_test)))


    ##### Save dataframe with results
    to_save = pd.DataFrame(
        {'Time': dFR_X_test.index,
         'Predictions': predictions,
         'Feinheit': Y_test})
    to_save = to_save.set_index('Time')
    to_save.to_csv(current_model)


    ##### Just for fun: using all X data and 10fold cross validation, calculate RMSE
    #estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=2)
    #kfold = KFold(n_splits=10, random_state=seed) #Nacho --> kein k-Fold
    #results = cross_val_score(estimator,X ,Y , cv=kfold)
    #print(filename_to_save + " with entire X/Y dataset and 10fold cross validation has RMSE of: " + str(math.sqrt(results.mean())))



    # Questions Nacho
    # 1) Correct to first compile model & then invoke cross_val_score, then fit & predict?
    # 2) Why are Models fitted with standardized X so much worse?
    # 3) Only standardize X? Not Y? What about the results?
    # 4) Standardize using Pipeline or Sebastian Raschka?
    # 5) MinMax Scaler

    # standardize dataset
    #std_scale = preprocessing.StandardScaler().fit(X)
    #X_std = std_scale.transform(X)
    #http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#standardization-and-min-max-scaling

    # evaluate model with standardized dataset
    #estimatorS = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
    #kfold = KFold(n_splits=10, random_state=seed)
    #resultsS = cross_val_score(estimatorS, X_std, Y, cv=kfold)
    #print("Standardized " + i + ": %.2f (%.2f) MSE" % (resultsS.mean(), resultsS.std()))

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

    # evaluate model with standardized dataset
    np.random.seed(seed)
    estimatorsS = []
    estimatorsS.append(('standardize', StandardScaler()))
    estimatorsS.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipelineS = Pipeline(estimatorsS)
    kfoldS = KFold(n_splits=10, random_state=seed)
    resultsS = cross_val_score(pipelineS, X, Y, cv=kfoldS)
    print("Standardized " + i + ": %.2f (%.2f) MSE" % (resultsS.mean(), resultsS.std()))

    pipelineS.fit(X, Y, batch_size=5, epochs=5, verbose=2, validation_split=0.2)

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