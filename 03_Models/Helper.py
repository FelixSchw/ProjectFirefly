# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import math
from sklearn.metrics import make_scorer, mean_squared_error

# Defition einer Error-Funktion (RMSE)
def errorFunction(y, y_pred):
    accuracy = math.sqrt(mean_squared_error(y, y_pred))
    return accuracy