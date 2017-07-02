# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import make_scorer, mean_squared_error

#Defition einer Error-Funktion (RMSE)
def errorFunction(y,y_pred):
    accuracy = math.sqrt(mean_squared_error(y, y_pred))
    return accuracy

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
filenames.append("RidgeSnapZeroResults.csv")
filenames.append("RidgeSnapLagResults.csv")
filenames.append("RidgeTimeSeriesCharacResults.csv")
filenames.append("RidgeARXResults.csv")
filenames.append("LassoSnapZeroResults.csv")
filenames.append("LassoSnapLagResults.csv")
filenames.append("LassoTimeSeriesCharacResults.csv")
filenames.append("LassoARXResults.csv")
filenames.append("SVRSnapZeroResults.csv")
filenames.append("SVRSnapLagResults.csv")
filenames.append("SVRTimeSeriesCharacResults.csv")
filenames.append("SVRARXResults.csv")
filenames.append("ANNSnapZeroResults.csv")
filenames.append("ANNSnapLagResults.csv")
filenames.append("ANNTimeSeriesCharacResults.csv")
#filenames.append("ANNSnapZeroResults.csv")

run_once = 0

## loop through all files
for file in filenames:
    dataForEvaluation = pd.read_csv(file, parse_dates=['Time'], date_parser=dateparse)
    dataForEvaluation = dataForEvaluation.set_index('Time')

    ###Berechnung des Errors wenn immer mean vorhergesagt wird
    if (run_once == 0):
        errorUsingMedian = errorFunction([np.mean(dataForEvaluation['Feinheit']) for i in range(0, len(dataForEvaluation['Feinheit']))], dataForEvaluation['Feinheit'])
        print("Error-Function of always predicting mean (", np.mean(dataForEvaluation['Feinheit']), ") : ", errorUsingMedian)
        run_once = 1

    ###Berechnung des Prediktion-Errors
    error = errorFunction(dataForEvaluation['Predictions'], dataForEvaluation['Feinheit'])
    print("Error-Function of", file, ": ", error)
