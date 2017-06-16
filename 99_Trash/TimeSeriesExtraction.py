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

###Only apply if default directories are not working###

###Change working directory###
f = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze"
l = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"

os.chdir(l)

###check if change of working directory worked###
cwd = os.getcwd()

#read the csv files and parse dates
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingData = pd.read_csv('Predictors.txt', parse_dates=['Time'], date_parser=dateparse)
trainingData = trainingData.set_index('Time')

trainingDataTargets = pd.read_csv('Targets.txt', parse_dates=['Time'], date_parser=dateparse)
trainingDataTargets = trainingDataTargets.set_index('Time')

###Drop irrelevant values###
del trainingData['Frischgut_Klinker_t/h']
del trainingData['Frischgut_Gips_t/h']
del trainingData['Frischgut_Huettensand_t/h']
del trainingData['Frischgut_Anhydrit_t/h']
del trainingData['Frischgut_Mahlhilfe_1_l/h']
del trainingData['Frischgut_Mahlhilfe_2_l/h']
del trainingData['Becherwerk_Strom_A']
del trainingData['Muehle_K1_Fuellstand_%']
del trainingData['Muehle_K2_Fuellstand_%']
del trainingData['Muehle_nach_Druck_mbar']
del trainingData['Filter_Ventilator_Strom_A']

###Multiindex-Dataframes die mit den Werten der Input-Parameter befüllt werden können###
ArrayAttributes = list(trainingData)
ArrayAttributesDelay = [20,20,20,3,0,0,0,0,2]
Array2Hours = [i for i in range(0,120)]
ArrayAmountOfTargets = [i for i in range(0,len(trainingDataTargets))]
ownIndex = pd.MultiIndex.from_product([ArrayAttributes, Array2Hours], names=['Attribute', '120Werte'])
TrainingDataAlloc = pd.DataFrame(index=ArrayAmountOfTargets, columns=ownIndex)
TrainingDataAllocSmall = pd.DataFrame(index=ArrayAmountOfTargets, columns=ArrayAttributes)


###Zuordnen der 120 Predikoren zu TrainingDataAlloc
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

TrainingDataAlloc = TrainingDataAlloc.set_index(trainingDataTargets.index)
TrainingDataAlloc = TrainingDataAlloc.stack()
TrainingDataAlloc = TrainingDataAlloc.reset_index(level=['Time', '120Werte'])

TrainingDataAlloc = TrainingDataAlloc.dropna()

extracted_features = extract_features(TrainingDataAlloc, column_id="Time", column_sort="120Werte")