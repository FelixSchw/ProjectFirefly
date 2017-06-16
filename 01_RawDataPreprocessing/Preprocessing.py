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

# Folgende Parameter inkludieren
#    - Muehle_nach_Temp._C mit lag=62min
#    - Griesse_t/h mit lag=10min
#    - Gesamtaufgabe_t/h mit lag=22min
#    - Frischgut_Summe_t/h mit lag=25min

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
lowerBoundary = [len(trainingData.columns)]
upperBoundary = [len(trainingData.columns)]
for j in range(0,len(trainingData.columns)):
    lowerBoundary = trainingData[trainingData.columns[j]].median() - 1.9 * trainingData[trainingData.columns[j]].std()
    upperBoundary = trainingData[trainingData.columns[j]].median() + 1.5 * trainingData[trainingData.columns[j]].std()
    print("Variable " + trainingData.columns[j] + " has lower Boundary of " + str(lowerBoundary) + " and upper Boundary of " + str(upperBoundary))


#Funktion um Outlier zu erkennen
def is_outlier(points, thresh = 3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    if(median != 0):
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
    elif(median == 0):
        modified_z_score = 0

    return modified_z_score > thresh

###Multiindex-Dataframes die mit den Werten der Input-Parameter befüllt werden können###
ArrayAttributes = list(trainingData)
Array2Hours = [i for i in range(0,120)]
ArrayAmountOfTargets = [i for i in range(0,len(trainingDataTargets))]
ownIndex = pd.MultiIndex.from_product([ArrayAttributes, Array2Hours], names=['Attribute', '120Werte'])
TrainingDataAlloc = pd.DataFrame(index=ArrayAmountOfTargets, columns=ownIndex)
TrainingDataAllocSmall = pd.DataFrame(index=ArrayAmountOfTargets, columns=ArrayAttributes)


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
                    if (trainingDataBuffer.iloc[k, j] > lowerBoundary) & (trainingDataBuffer.iloc[k, j] < upperBoundary):
                        TrainingDataAlloc.ix[i, (trainingData.columns[j], k)] = trainingDataBuffer.iloc[k, j]
                    else:
                        print("Variable " + trainingData.columns[j] + " mit Median " + str(
                            trainingData[trainingData.columns[j]].median()) + " und Stabw " + str(
                            trainingData[trainingData.columns[j]].std()) + ": Der Wert " + str(
                            trainingDataBuffer.iloc[k, j]) + " fliegt raus!")
                        TrainingDataAlloc.ix[i, (trainingData.columns[j], k)] = trainingData[trainingData.columns[j]].median()

###Outlier Detection and Removal

os.chdir(pathInterface)
cwd = os.getcwd()

TrainingDataAlloc.to_csv("PreprocessedPredictors.csv")
