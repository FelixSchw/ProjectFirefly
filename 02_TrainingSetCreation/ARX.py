# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os


###Only apply if default directories are not working###

###Change working directory###
from sklearn.preprocessing import StandardScaler

felixOrLeo = "l"

if (felixOrLeo == "f"):
    pathData = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze"
    pathInterface = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Interface"
else:
    pathData = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"
    pathInterface = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Interface"

### extract y values
os.chdir(pathData)
cwd = os.getcwd()
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingDataTargets = pd.read_csv('Targets.txt', parse_dates=['Time'], date_parser=dateparse)
trainingDataTargets = trainingDataTargets.set_index('Time')

### extract 'TrainingDataAlloc' from previous module 'Preprocessing'
os.chdir(pathInterface)
cwd = os.getcwd()
trainingDataPredictors = pd.read_csv('PreprocessedPredictors.csv', header=[0, 1], skipinitialspace=True, tupleize_cols=True)
trainingDataPredictors = trainingDataPredictors.drop(trainingDataPredictors.columns[0], axis=1)
trainingDataPredictors.columns = pd.MultiIndex.from_tuples(trainingDataPredictors.columns, names=['Attribute', '120Werte'])
trainingDataPredictors.columns = trainingDataPredictors.columns.set_levels(trainingDataPredictors.columns.levels[1].astype(int), level=1)

### create dataframe for predictor snapshot
ArrayAmountOfTargets = [i for i in range(0,len(trainingDataTargets))]
ArrayAttributes = trainingDataPredictors.columns.levels[0]
CustomColumnNames = []

### add column with autoregressive values at t-1
CustomColumnNames.append("Feinheit_t-1")

for i in range(0, len(ArrayAttributes)):
    for j in range(0, 120):
        CustomColumnNames.append(ArrayAttributes[i] + "_" + str(j))

### create dataframe with all ARX
predictorsSnapshot = pd.DataFrame(index=ArrayAmountOfTargets, columns=CustomColumnNames)

### fill in autoregressive first column
y_values = trainingDataTargets.values
y_values_last_removed = np.delete(y_values, len(y_values)-1)
to_add = []
to_add.append(y_values[0][0])
y_tminusone = np.concatenate((to_add, y_values_last_removed))
predictorsSnapshot["Feinheit_t-1"] = y_tminusone


### fill in x columns with time lags
for i in range(0, len(ArrayAttributes)):
    for j in range(0, 120):
        predictorsSnapshot[(ArrayAttributes[i] + "_" + str(j))] = trainingDataPredictors[ArrayAttributes[i]][j]

###Zusammenfügen Prediktoren und Target
predictorsSnapshot = predictorsSnapshot.set_index(trainingDataTargets.index)
dataForRegression = pd.concat([predictorsSnapshot, trainingDataTargets], axis=1, join_axes=[trainingDataTargets.index])

#Löschen der Spalten mit Null-Werten
dataForRegression = dataForRegression.dropna()

#Konvertieren in float64 dtype
for j in range(0, len(ArrayAttributes)):
    for k in range(0,120):
        dataForRegression.ix[:, (ArrayAttributes[j] + "_" + str(k))] = pd.to_numeric(dataForRegression[(ArrayAttributes[j] + "_" + str(k))])

dataForRegression.to_csv("ARX.csv")