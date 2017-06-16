# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

###Only apply if default directories are not working###

###Change working directory###
from sklearn.preprocessing import StandardScaler

felixOrLeo = "f"

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
trainingDataPredictors = pd.read_csv('TrainingDataAlloc.csv', header=[0, 1], skipinitialspace=True, tupleize_cols=True)
trainingDataPredictors = trainingDataPredictors.drop(trainingDataPredictors.columns[0], axis=1)
trainingDataPredictors.columns = pd.MultiIndex.from_tuples(trainingDataPredictors.columns, names=['Attribute', '120Werte'])
trainingDataPredictors.columns = trainingDataPredictors.columns.set_levels(trainingDataPredictors.columns.levels[1].astype(int), level=1)

#Restack dataframe according to format of tsfresh
trainingDataPredictors = trainingDataPredictors.set_index(trainingDataTargets.index)
trainingDataPredictors = trainingDataPredictors.stack()
trainingDataPredictors = trainingDataPredictors.reset_index(level=['Time', '120Werte'])
trainingDataPredictors = trainingDataPredictors.dropna()

#Extract features using tsfresh
extracted_features = extract_features(trainingDataPredictors, column_id="Time", column_sort="120Werte")

#Select features
impute(extracted_features)
features_filtered = select_features(extracted_features, trainingDataTargets.iloc[:,0])


#Write data to csv for prediction
os.chdir(pathInterface)
cwd = os.getcwd()

features_filtered.to_csv("timeSeriesCharac.csv")