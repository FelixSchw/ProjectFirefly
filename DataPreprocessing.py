# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import array
import os
from scipy.stats import skew

###Only apply if default directories are not working###

###Change working directory###
f = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze\\12-05-2017\\Fertige_Sets"
l = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze/12-05-2017/Fertige_Sets"
os.chdir(l)

###check if change of working directory worked###
cwd = os.getcwd()

#read the csv files and parse dates
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingData = pd.read_csv('Predictors_1_mit_Testdaten.csv', parse_dates=['Time'], date_parser=dateparse)
trainingData = trainingData.set_index('Time')

trainingDataTargets = pd.read_csv('Targets_1.csv', parse_dates=['Time'], date_parser=dateparse)
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
del trainingData['Frischgut_Gips._t/h']
del trainingData['Frischgut_Huettensand_t/h']
del trainingData['Frischgut_Anhydrit_t/h']
del trainingData['Frischgut_Mahlhilfe 1_l/h']
del trainingData['Frischgut_Mahlhilfe 2_l/h']
del trainingData['Becherwerk_Strom_A']
del trainingData['Muehle_K1_Fuellstand_%']
del trainingData['Muehle_K2_Fuellstand_%']
del trainingData['Muehle_nach_Druck_mbar']
del trainingData['Filter_Ventilator_Strom_A']

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

###Multiindex-Dataframe der mit den Werten der Input-Parameter befüllt werden kann###
ArrayAttributes = list(trainingData)
Array2Hours = [i for i in range(0,120)]
ArrayAmountOfTargets = [i for i in range(0,len(trainingDataTargets))]
ownIndex = pd.MultiIndex.from_product([ArrayAttributes, Array2Hours], names=['Attribute', '120Werte'])
TrainingDataAlloc = pd.DataFrame("NaN", index=ArrayAmountOfTargets, columns=ownIndex)



###Zuordnen der 120 Predikoren zu dem jeweiligen Target
for i in range(0, len(trainingDataTargets)):
    startTime = trainingDataTargets.index[i] - pd.Timedelta(minutes=120)
    endTime = trainingDataTargets.index[i]
    trainingDataBuffer = trainingData.loc[(trainingData.index >= startTime) & (trainingData.index <= endTime), :]
    # Nur Targets mit 120 Messungen verwenden
    if (len(trainingDataBuffer)>= 120):
        for j in range(0, len(trainingData.columns)):
            for k in range(0,120):
                #Einfügen in Zeile i und Spalte j (mit Unterspalte k)
                TrainingDataAlloc.ix[i, (trainingData.columns[j],k)] = trainingDataBuffer.iloc[k,j]

print(TrainingDataAlloc)