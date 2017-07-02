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
predictorsSnapshot = pd.DataFrame(index=ArrayAmountOfTargets, columns=ArrayAttributes)

### create snapLag.csv
ArrayAttributesDelay = [25,10,22,3,0,0,0,0,62]
###Zuordnen 1 Prediktor jedes Attributs zu TrainingDataAllocSmall
for i in range(0, len(trainingDataTargets)):
    for j in range(0, len(ArrayAttributes)):
        predictorsSnapshot.loc[i,ArrayAttributes[j]] = trainingDataPredictors.ix[i, ArrayAttributes[j]][ArrayAttributesDelay[j]]

###Zusammenfügen Prediktoren und Target
predictorsSnapshot = predictorsSnapshot.set_index(trainingDataTargets.index)
dataForRegression = pd.concat([predictorsSnapshot, trainingDataTargets], axis=1, join_axes=[trainingDataTargets.index])

#Löschen der Spalten mit Null-Werten
dataForRegression = dataForRegression.dropna()

#Konvertieren in float64 dtype
for j in range(0, len(ArrayAttributes)):
    dataForRegression.ix[:, ArrayAttributes[j]] = pd.to_numeric(dataForRegression[ArrayAttributes[j]])

dataForRegression.to_csv("SnapLag.csv")