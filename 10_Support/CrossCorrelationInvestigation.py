# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import array
import os
from scipy.stats import skew
from sklearn import cross_validation, linear_model
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import math
from sklearn import preprocessing
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from pandas.plotting import autocorrelation_plot


###Only apply if default directories are not working###

###Change working directory###
f = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze\\26-05-2017"
l = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"

os.chdir(l)

##### Check if change of working directory worked###
cwd = os.getcwd()


##### Read the csv files and parse dates
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingData = pd.read_csv('Predictors.txt', parse_dates=['Time'], date_parser=dateparse)
trainingData = trainingData.set_index('Time')
trainingDataTargets = pd.read_csv('Targets.txt', parse_dates=['Time'], date_parser=dateparse)
trainingDataTargets = trainingDataTargets.set_index('Time')


##### Quick Data Exploration
print('--------------Shape----------------')
print(trainingData.shape)
print('--------------Describe----------------')
print(trainingData.describe())
print('--------------IsNull----------------')
print(pd.isnull(trainingData).sum())
print('--------------Pearson Corr----------------')
print(trainingData.corr(method='pearson'))


##### Multiindex-Dataframes die mit den Werten der Input-Parameter befüllt werden können###
ArrayAttributes = list(trainingData)
ArrayAttributesDelay = [20,20,20,3,0,0,0,0,2]
Array2Hours = [i for i in range(0,120)]
ArrayAmountOfTargets = [i for i in range(0,len(trainingDataTargets))]
ownIndex = pd.MultiIndex.from_product([ArrayAttributes, Array2Hours], names=['Attribute', '120Werte'])
TrainingDataAlloc = pd.DataFrame(index=ArrayAmountOfTargets, columns=ownIndex)
TrainingDataAllocSmall = pd.DataFrame(index=ArrayAmountOfTargets, columns=ArrayAttributes)


##### Zuordnen der 120 Predikoren zu TrainingDataAlloc
for i in range(0, len(trainingDataTargets)):
#    if (trainingDataTargets.loc[(trainingDataTargets.index[5]),"Feinheit"] > 0):
        startTime = trainingDataTargets.index[i] - pd.Timedelta(minutes=120)
        endTime = trainingDataTargets.index[i]
        trainingDataBuffer = trainingData.loc[(trainingData.index >= startTime) & (trainingData.index <= endTime), :]
        # Nur Targets mit 120 Messungen verwenden
        if (len(trainingDataBuffer)>= 120):
            # Nur Targets mit Labormesswerten >0 verwenden
            if (trainingDataTargets.ix[i,0] > 0):
                for j in range(0, len(trainingData.columns)):
                    for k in range(0,120):
                        #Einfügen in Zeile i und Spalte j (mit Unterspalte k)
                        TrainingDataAlloc.ix[i, (trainingData.columns[j],k)] = trainingDataBuffer.iloc[k,j]
ArrayWithHundredEntries = [i for i in range(0,119)]
CrossCorrelations = pd.DataFrame(index=ArrayWithHundredEntries, columns=ArrayAttributes)


##### Construct dataForRegression (essentially not necessary here) and CrossCorrelations
for h in range(0, len(ArrayWithHundredEntries)):
    for i in range(0, len(trainingDataTargets)):
        # print("i = " + str(i))
        for j in range(0, len(trainingData.columns)):
            # print("j = " + str(j))
            TrainingDataAllocSmall.ix[i,ArrayAttributes[j]] = TrainingDataAlloc.ix[i, (ArrayAttributes[j],119-ArrayWithHundredEntries[h])]

    ##### Zusammenfügen Prediktoren und Target
    TrainingDataAllocSmall = TrainingDataAllocSmall.set_index(trainingDataTargets.index)
    dataForRegression = pd.concat([TrainingDataAllocSmall, trainingDataTargets], axis=1, join_axes=[trainingDataTargets.index])
    ##### Löschen der Spalten mit Null-Werten
    dataForRegression = dataForRegression.dropna()

    # convert columns in object type to float64 type (dataForRegression.dtypes prints them out)
    for k in range(0, len(trainingData.columns)):
        dataForRegression.ix[:,trainingData.columns[k]] = pd.to_numeric(dataForRegression[trainingData.columns[k]])

    for g in range(0, len(trainingData.columns)):
        CrossCorrelations.ix[h, trainingData.columns[g]] = dataForRegression[trainingData.columns[g]].corr(dataForRegression['Feinheit'])


##### Save CrossCorrelations
CrossCorrelations.to_csv("CrossCorrelations.csv")


##### AutoCorrelation plot
autocorrelation_plot(trainingDataTargets.Feinheit)






