# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os


###Only apply if default directories are not working###

###Change working directory###
from sklearn.preprocessing import StandardScaler


##### Change directory to Felix or Leos computer
felixOrLeo = "l"
if (felixOrLeo == "f"):
    pathData = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze"
    pathInterface = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Interface"
else:
    pathData = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"
    pathInterface = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Interface"


##### Change directory
os.chdir(pathData)
cwd = os.getcwd()


##### Read the csv target and regressor data files and parse dates
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingData = pd.read_csv('Predictors.txt', parse_dates=['Time'], date_parser=dateparse)
trainingData = trainingData.set_index('Time')
trainingDataTargets = pd.read_csv('Targets.txt', parse_dates=['Time'], date_parser=dateparse)
trainingDataTargets = trainingDataTargets.set_index('Time')


##### Data Exploration
# print('--------------Shape----------------')
# print(trainingData.shape)
# print('--------------Describe----------------')
# print(trainingData.describe())
# print('--------------IsNull----------------')
# print(pd.isnull(trainingData).sum())
# print('--------------Pearson Corr----------------')
# print(trainingData.corr(method='pearson'))


##### Drop variables not considered to be relevant
del trainingData['Frischgut_Klinker_t/h']
del trainingData['Frischgut_Gips_t/h']
del trainingData['Frischgut_Huettensand_t/h']
del trainingData['Frischgut_Anhydrit_t/h']
del trainingData['Frischgut_Mahlhilfe_1_l/h']
del trainingData['Frischgut_Mahlhilfe_2_l/h']
del trainingData['Muehle_Strom_A']
del trainingData['Muehle_K1_Fuellstand_%']
del trainingData['Muehle_K2_Fuellstand_%']
del trainingData['Muehle_nach_Druck_mbar']
del trainingData['Filter_Ventilator_Strom_A']
del trainingData['Frischgut_Eisensulfat_kg/h']
del trainingData['Frischgut_Zinnsulfat_kg/h']


##### Multiindex-Dataframes die mit den Werten der Input-Parameter befüllt werden können###
ArrayAttributes = list(trainingData)
Array2Hours = [i for i in range(0,120)]
ArrayAmountOfTargets = [i for i in range(0,len(trainingDataTargets))]
ArrayAmountOfVariableObservations = [i for i in range(0,len(trainingDataTargets)*120)]
ownIndex = pd.MultiIndex.from_product([ArrayAttributes, Array2Hours], names=['Attribute', '120Werte'])
TrainingDataAlloc = pd.DataFrame(index=ArrayAmountOfTargets, columns=ownIndex)
TrainingDataAllocSmall = pd.DataFrame(index=ArrayAmountOfTargets, columns=ArrayAttributes)
VariablesDataframe = pd.DataFrame(index=ArrayAmountOfVariableObservations, columns=ArrayAttributes)


##### Zuordnen der 120 Predikoren zu TrainingDataAlloc
for i in range(0, len(trainingDataTargets)):
    if ((i % 5) == 0):
        print("Progress of Matrix Fill-in: " + str((i/len(trainingDataTargets))*100) + " %")
    startTime = trainingDataTargets.index[i] - pd.Timedelta(minutes=120)
    endTime = trainingDataTargets.index[i]
    trainingDataBuffer = trainingData.loc[(trainingData.index >= startTime) & (trainingData.index <= endTime), :]
    if (len(trainingDataBuffer)>= 120): # Nur Targets mit 120 Messungen verwenden
        if (trainingDataTargets.ix[i,0] > 0): # Nur Targets mit Labormesswerten >0 verwenden
            for j in range(0, len(trainingData.columns)):
                for k in range(0,120):
                    TrainingDataAlloc.ix[i, (trainingData.columns[j], k)] = trainingDataBuffer.iloc[k, j]


##### Calculate median and std of all variables
for i in range(0, len(ArrayAttributes)):
    columnsCollector = TrainingDataAlloc[ArrayAttributes[i]][0]
    for j in range(1,120):
        columnsCollector = columnsCollector.append(TrainingDataAlloc[ArrayAttributes[i]][j]).reset_index(drop=True)
    VariablesDataframe[ArrayAttributes[i]] = columnsCollector


##### Calculate upper and lower Tolerances for Outlier Removal
lowerTolerances = VariablesDataframe.median() - 3*VariablesDataframe.std()
upperTolerances = VariablesDataframe.median() + 3*VariablesDataframe.std()
lowerTolerances['Griesse_t/h'] = 5


##### Remove outliers
counterGood = 0
counterBad = 0
for i in range(0, len(trainingDataTargets)):
    if ((i % 5) == 0):
        print("######################################################################")
        print("Progress of Outlier Detection & Replacement: " + str((i/len(trainingDataTargets))*100) + " %")
        print("######################################################################")
    for j in range(0, len(trainingData.columns)):
        for k in range(0, 120):
            temp = TrainingDataAlloc.ix[i, (trainingData.columns[j], k)]
            if ((temp > lowerTolerances[ArrayAttributes[j]]) & (temp < upperTolerances[ArrayAttributes[j]])):
                counterGood = counterGood + 1
            else:
                TrainingDataAlloc.ix[i, (trainingData.columns[j], k)] = VariablesDataframe.median()[ArrayAttributes[j]]
                counterBad = counterBad + 1
                print("Variable " + trainingData.columns[j] + " mit Median " + str(
                    VariablesDataframe.median()[ArrayAttributes[j]]) + " und Stabw " + str(
                    VariablesDataframe.std()[ArrayAttributes[j]]) + ": Der Wert " + str(
                    temp) + " wird durch Median ersetzt")
print("Data Preprocessing and Outlier Detection finished. In total, " + str(counterGood) + " values were classified as OK and " + str(counterBad) + " values were classified as outliers")


##### Write TrainingDataAlloc to csv into interface directory
os.chdir(pathInterface)
cwd = os.getcwd()
TrainingDataAlloc.to_csv("PreprocessedPredictors.csv")