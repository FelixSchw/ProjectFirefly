# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


###Only apply if default directories are not working###

###Change working directory###
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

##### Import files from correct directory
felixOrLeo = "l"
if (felixOrLeo == "f"):
    pathData = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze"
    pathInterface = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Interface"
else:
    pathData = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"
    pathInterface = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Interface"


##### Extract y values
os.chdir(pathData)
cwd = os.getcwd()
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingDataTargets = pd.read_csv('Targets.txt', parse_dates=['Time'], date_parser=dateparse)
trainingDataTargets = trainingDataTargets.set_index('Time')


##### Extract 'TrainingDataAlloc' from previous module 'Preprocessing'
os.chdir(pathInterface)
cwd = os.getcwd()
trainingDataPredictors = pd.read_csv('PreprocessedPredictors.csv', header=[0, 1], skipinitialspace=True, tupleize_cols=True)
trainingDataPredictors = trainingDataPredictors.drop(trainingDataPredictors.columns[0], axis=1)
trainingDataPredictors.columns = pd.MultiIndex.from_tuples(trainingDataPredictors.columns, names=['Attribute', '120Werte'])
trainingDataPredictors.columns = trainingDataPredictors.columns.set_levels(trainingDataPredictors.columns.levels[1].astype(int), level=1)
ArrayAttributes = trainingDataPredictors.columns.levels[0]


##### Create u, the matrix with lagged explanatory variables
steps = [40, 60, 80, 100, 119]
temp_list_with_columns = []
for i in range(0, len(ArrayAttributes)):
    for j in range(0, len(steps)):
        temp_list_with_columns.append(trainingDataPredictors[ArrayAttributes[i]][steps[j]].values)
x = np.vstack(temp_list_with_columns).T


##### Create Y
y = trainingDataTargets.values


##### Standardize data und split into test and train
test_size = 0.2
seed = 1
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y, test_size=test_size, random_state=seed)


##### Compile ARMAX model
armax = sm.tsa.ARMA(Y_train, order=(1, 1), exog=X_train).fit()


##### Im array u sind noch nuller und ausreißer - um diese zu finden
#np.savetxt("NullerAusreisserfuerARMAX.csv", u, delimiter=",")


##### Prediction noch zu machen
armax