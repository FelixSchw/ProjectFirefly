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


### change directory ###
os.chdir(pathData)
cwd = os.getcwd()

#read the csv files and parse dates
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingData = pd.read_csv('Predictors.txt', parse_dates=['Time'], date_parser=dateparse)

#extracted_features = extract_features(trainingData, column_id="", column_sort="Time")

trainingData = trainingData.set_index('Time')

trainingDataTargets = pd.read_csv('Targets.txt', parse_dates=['Time'], date_parser=dateparse)
trainingDataTargets = trainingDataTargets.set_index('Time')

# ###Data Exploration###
# print('--------------Shape----------------')
# print(trainingData.shape)
# print('--------------Describe----------------')
# print(trainingData.describe())
# print('--------------IsNull----------------')
# print(pd.isnull(trainingData).sum())
# print('--------------Pearson Corr----------------')
# print(trainingData.corr(method='pearson'))

###Drop irrelevant values###
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

### Outlier interval definition
####### not optimal as this uses median and std from data that also includes production of other mixtures ###########
OneEntryArray = [i for i in range(0,1)]
lowerBoundary = pd.DataFrame(index=OneEntryArray, columns=trainingData.columns)
upperBoundary = pd.DataFrame(index=OneEntryArray, columns=trainingData.columns)

for j in range(0,len(trainingData.columns)):
    lowerBoundary[trainingData.columns[j]] = trainingData[trainingData.columns[j]].median() - 1.9 * trainingData[trainingData.columns[j]].std()
    upperBoundary[trainingData.columns[j]] = trainingData[trainingData.columns[j]].median() + 1.5 * trainingData[trainingData.columns[j]].std()
    print("Variable " + trainingData.columns[j] + " has lower Boundary of " + str(lowerBoundary) + " and upper Boundary of " + str(upperBoundary))


###Multiindex-Dataframes die mit den Werten der Input-Parameter befüllt werden können###
ArrayAttributes = list(trainingData)
Array2Hours = [i for i in range(0,120)]
ArrayAmountOfTargets = [i for i in range(0,len(trainingDataTargets))]
ownIndex = pd.MultiIndex.from_product([ArrayAttributes, Array2Hours], names=['Attribute', '120Werte'])
TrainingDataAlloc = pd.DataFrame(index=ArrayAmountOfTargets, columns=ownIndex)
TrainingDataAllocSmall = pd.DataFrame(index=ArrayAmountOfTargets, columns=ArrayAttributes)

changesCounter = 0

###Zuordnen der 120 Predikoren zu TrainingDataAlloc
for i in range(0, len(trainingDataTargets)):
    if ((i % 5) == 0):
        print("Progress: " + str((i/len(trainingDataTargets))*100) + " %")
    startTime = trainingDataTargets.index[i] - pd.Timedelta(minutes=120)
    endTime = trainingDataTargets.index[i]
    trainingDataBuffer = trainingData.loc[(trainingData.index >= startTime) & (trainingData.index <= endTime), :]
    # Nur Targets mit 120 Messungen verwenden
    if (len(trainingDataBuffer)>= 120):
        # Nur Targets mit Labormesswerten >0 verwenden
        if (trainingDataTargets.ix[i,0] > 0):
            for j in range(0, len(trainingData.columns)):
                for k in range(0,120):
                    # Einfügen in Zeile i und Spalte j (mit Unterspalte k)
                    if ((trainingDataBuffer.iloc[k, j] > lowerBoundary[trainingData.columns[j]][0]) & (trainingDataBuffer.iloc[k, j] < upperBoundary[trainingData.columns[j]][0])):
                        TrainingDataAlloc.ix[i, (trainingData.columns[j], k)] = trainingDataBuffer.iloc[k, j]
                    else:
                        changesCounter = changesCounter + 1
                        print("Variable " + trainingData.columns[j] + " mit Median " + str(
                            trainingData[trainingData.columns[j]].median()) + " und Stabw " + str(
                            trainingData[trainingData.columns[j]].std()) + ": Der Wert " + str(
                            trainingDataBuffer.iloc[k, j]) + " wird durch Median ersetzt")
                        TrainingDataAlloc.ix[i, (trainingData.columns[j], k)] = trainingData[trainingData.columns[j]].median()

print("In total " + str(changesCounter) + " datapoints were detected as outliers and replaced by median")

os.chdir(pathInterface)
cwd = os.getcwd()

TrainingDataAlloc.to_csv("PreprocessedPredictors.csv")