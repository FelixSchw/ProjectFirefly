# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os
from scipy.stats import skew

###Only apply if default directories are not working###

###Change working directory###
f = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze\\12-05-2017\\Fertige_Sets"
l = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze/12-05-201/Fertige_Sets//"
os.chdir(f)

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

startDate = str(trainingDataTargets.index.get_values()[0])[:10]
endDate = str(trainingDataTargets.index.get_values()[0])[:10]
startTime = str(trainingDataTargets.index.get_values()[i] - pd.Timedelta(minutes=120))[11:19]
endTime = str(trainingDataTargets.index.get_values()[0])[11:19]
trainingDataBuffer = trainingData[startDate:endDate].between_time(start_time= startTime, end_time = endTime)

for i in range(0, len(trainingDataTargets)):
    startDate = str(trainingDataTargets.index.get_values()[i])[:10]
    endDate = str(trainingDataTargets.index.get_values()[i])[:10]
    startTime = str(trainingDataTargets.index.get_values()[i] - pd.Timedelta(minutes=120))[11:19]
    endTime = str(trainingDataTargets.index.get_values()[i])[11:19]
    trainingDataBuffer = trainingData[startDate:endDate].between_time(start_time=startTime, end_time=endTime)
    trainingDataBuffer = trainingDataBuffer.unstack()

# for i in range(0, len(trainingDataTargets)):
#     for j in range(0, len(trainingData.columns)):
#         #for k in range (0,120):
#             trainingDataAllocated = pd.concat(trainingData.iloc[1,j])