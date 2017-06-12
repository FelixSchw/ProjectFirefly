# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import math
import array
import os
#from scipy.stats import skew
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


###Only apply if default directories are not working###

###Change working directory###
f = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze\\26-05-2017"
l = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze"

os.chdir(l)

###check if change of working directory worked###
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


#Defition einer Error-Funktion (RMSE)
def errorFunction(y,y_pred):
    accuracy = math.sqrt(mean_squared_error(y, y_pred))
    return accuracy

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
ArrayAttributesDelay = [25,10,22,3,0,0,0,0,62]
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


###Zuordnen 1 Prediktor jedes Attributs zu TrainingDataAllocSmall
for i in range(0, len(trainingDataTargets)):
    for j in range(0, len(trainingData.columns)):
        TrainingDataAllocSmall.loc[i,ArrayAttributes[j]] = TrainingDataAlloc.ix[i, (ArrayAttributes[j],119-ArrayAttributesDelay[j])]

###Zusammenfügen Prediktoren und Target
TrainingDataAllocSmall = TrainingDataAllocSmall.set_index(trainingDataTargets.index)
dataForRegression = pd.concat([TrainingDataAllocSmall, trainingDataTargets], axis=1, join_axes=[trainingDataTargets.index])

#Löschen der Spalten mit Null-Werten
dataForRegression = dataForRegression.dropna()

#Konvertieren in float64 dtype
for j in range(0, len(trainingData.columns)):
    dataForRegression.ix[:, trainingData.columns[j]] = pd.to_numeric(dataForRegression[trainingData.columns[j]])

#Vorgehen Tutorial
# load dataset
#dataframe = pd.read_csv("/Users/leopoldspenner/Documents/python/housing.csv", delim_whitespace=True, header=None)
#datasett = dataframe.values
# split into input (X) and output (Y) variables
#XX = datasett[:,0:13]
#YY = datasett[:,13]

#Standardisieren der Trainingsdaten
#dataForRegression_X = pd.DataFrame(preprocessing.scale(dataForRegression_X))
#dataForRegression_X = dataForRegression_X.set_index(dataForRegression.ix[:,:len(trainingData.columns)].index)


dataset = dataForRegression.values
X = dataset[:,0:9]
Y = dataset[:,9]

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator,X ,Y , cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
