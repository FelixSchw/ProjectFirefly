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
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import math

###Only apply if default directories are not working###

###Change working directory###
f = "C:\\Users\\Felix Schweikardt\\Dropbox\\Seminararbeit FZI - Softsensor\\Datensätze\\26-05-2017"
l = "/Users/leopoldspenner/Dropbox/Seminararbeit FZI - Softsensor/Datensätze/26-05-2017"

os.chdir(l)

###check if change of working directory worked###
cwd = os.getcwd()

#read the csv files and parse dates
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
trainingData = pd.read_csv('Predictors_3.txt', parse_dates=['Time'], date_parser=dateparse)
trainingData = trainingData.set_index('Time')

trainingDataTargets = pd.read_csv('Targets_3.txt', parse_dates=['Time'], date_parser=dateparse)
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


###Zuordnen 1 Prediktor jedes Attributs zu TrainingDataAllocSmall
for i in range(0, len(trainingDataTargets)):
    for j in range(0, len(trainingData.columns)):
        TrainingDataAllocSmall.loc[i,ArrayAttributes[j]] = TrainingDataAlloc.ix[i, (ArrayAttributes[j],119-ArrayAttributesDelay[j])]

###Zusammenfügen Prediktoren und Target
TrainingDataAllocSmall = TrainingDataAllocSmall.set_index(trainingDataTargets.index)
dataForRegression = pd.concat([TrainingDataAllocSmall, trainingDataTargets], axis=1, join_axes=[trainingDataTargets.index])

#Löschen der Spalten mit Null-Werten
dataForRegression = dataForRegression.dropna()

#Defition einer Error-Funktion
def errorFunction(y,y_pred):
    accuracy = r2_score(y, y_pred)
    return accuracy

#Aufteilen von  trainingData in Subsets von Trainings- und "Test"-Trainingsdaten mit Parametern seed & test_size
seed = 1
test_size = 0.25
X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataForRegression.ix[:,:len(trainingData.columns)], dataForRegression.ix[:,'Feinheit':], test_size=test_size, random_state=seed)

#Defintion verschiedener Modelle
linReg = linear_model.LinearRegression(n_jobs=-1)
ridge = linear_model.Ridge(alpha=1)
lasso = linear_model.Lasso(alpha=0.005)
svr_rbf = SVR(kernel='rbf', C=100000, gamma=1e-7)
svr_lin = SVR(kernel='linear', C=0.0001)
svr_poly = SVR(kernel='poly', C=100000, degree=5, epsilon=1)

#Auswahl des Modells
clf1 = ridge

#training of classifier
clf1.fit(X_train, y_train)

#Prediction des Subsets von "Test"-Trainingsdaten mit clf1
prediction_clf1 = pd.DataFrame(clf1.predict(X_test))
prediction_clf1 = prediction_clf1.set_index(X_test.index)
prediction_clf1.columns = ['Predictions']
prediction_clf1_solution = pd.concat([X_test, prediction_clf1, y_test], axis=1, join_axes=[X_test.index])
print("Prediction using (clf1): ")
print(prediction_clf1_solution)

# ###Berechnung des Prediktion-Errors
# error_clf1 = clf1.score(X_test, y_test)
# print("R^2 of chosen regression (clf1): ", error_clf1)
# errorFunction_clf1 = errorFunction(prediction_clf1, y_test)
# print("Error-Function of chosen regression (clf1): ", errorFunction_clf1)

# #Initialisierung der Error-Funktion
# scorer = make_scorer(score_func=errorFunction, greater_is_better=True)
#
# #Cross Validation von Ridge Parameters
# if clf1._get_param_names() == linear_model.Ridge()._get_param_names():
#     alphas = np.array([1e-15, 1e-10, 1e-5, 0.001, 0.1, 1, 100, 1000, 10000])
#     alphas_grid = dict(alpha=alphas)
#     clf_ridge = linear_model.Ridge()
#     grid = GridSearchCV(estimator=clf_ridge, param_grid=alphas_grid, cv=2, scoring=scorer)
#     grid.fit(TrainingDataAllocSmall, trainingDataTargets)
#     print("\nRSME k-folded (k=10) Ridge-Regressions with different alpha:")
#     print(*grid.grid_scores_, sep="\n")