# -*- coding: utf-8 -*-

#author: Schweikardt & Spenner
#version: 1.0 May 2017

import pandas as pd
import numpy as np
import os
import Helper as hlpr
import matplotlib.pyplot as plt
#import seaborn as sns

#Fetching training data set
felixOrLeo = "f"
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
#filenames.append("ANNSnapZeroResults.csv")
#filenames.append("ANNSnapLagResults.csv")
#filenames.append("ANNTimeSeriesCharacResults.csv")
#filenames.append("ANNSnapZeroResults.csv")

toleranceInterval = 0.325
evaluationResults = pd.DataFrame(columns=('Method', 'ValueOf_RMSE', 'PercentageInTolorance'))
run_once = 0
plotCounter = 3

## loop through all files
for file in filenames:
    dataForEvaluation = pd.read_csv(file, parse_dates=['Time'], date_parser=dateparse)
    dataForEvaluation = dataForEvaluation.set_index('Time')

    ###Berechnung des Errors wenn immer mean vorhergesagt wird
    if (run_once == 0):
        print("### General information for better understanding of Evaluation ###")
        print("The set tolerance interval is: ", toleranceInterval)
        rangeOfData = dataForEvaluation['Feinheit'].max() - dataForEvaluation['Feinheit'].min()
        print("The range (Max-Min) of the values Feinheit is: ", rangeOfData)
        errorUsingMean = hlpr.errorFunction([np.mean(dataForEvaluation['Feinheit']) for i in range(0, len(dataForEvaluation['Feinheit']))], dataForEvaluation['Feinheit'])
        print("Error-Function of always predicting mean (", np.mean(dataForEvaluation['Feinheit']), ") : ", errorUsingMean, "\n")
        run_once = 1

    ###Berechnung des Prediktion-Errors
    error = hlpr.errorFunction(dataForEvaluation['Predictions'], dataForEvaluation['Feinheit'])

    ###Berechnung der Prozentzahl in Toleranz-Bereich
    dataForEvaluation['inTolerance'] = np.where(abs(dataForEvaluation['Predictions']-dataForEvaluation['Feinheit']) <= toleranceInterval, 'Good', 'Bad')
    amountGood = dataForEvaluation['inTolerance'].value_counts().loc['Good']
    amountBad = dataForEvaluation['inTolerance'].value_counts().loc['Bad']
    percentInTolerance = amountGood / (amountGood+amountBad)

    ###Speichern in Dataframe
    evaluationResults.loc[len(evaluationResults)] = [file, error, percentInTolerance]

    ###Plot Diagramms
    dataForPlot = dataForEvaluation[['Feinheit', 'Predictions']]
    dataForPlot = dataForPlot.sort_values('Feinheit')
    dataForPlot['Index'] = [i for i in range(0, len(dataForEvaluation))]

    plotCounter += 1
    if (plotCounter > 3):
        f, axarr = plt.subplots(4, sharex=True)
        #f.suptitle('Visualizierung der Evaluation')
        plotCounter = 0

    RMSEString = "%1.3f" %evaluationResults.iloc[evaluationResults['Method'][evaluationResults['Method']==file].index,1]
    PercentageString = "%1.3f" % evaluationResults.iloc[evaluationResults['Method'][evaluationResults['Method'] == file].index, 2]
    axarr[plotCounter].set_title(file + "(" + RMSEString + " RMSE/ " + PercentageString + " InClass)")
    axarr[plotCounter].scatter(dataForPlot['Index'], dataForPlot['Feinheit'], color='#1f77b4')
    axarr[plotCounter].scatter(dataForPlot['Index'], dataForPlot['Predictions'], color='#d62728')
    axarr[plotCounter].errorbar(dataForPlot['Index'], dataForPlot['Predictions'], yerr=toleranceInterval, color='#d62728',
                                ecolor='r', fmt='o', capsize=5)

    #plt.savefig('testEvaluation.png', bbox_inches='tight')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


##Ausgabe der Ergebnisse
print(evaluationResults)
evaluationResults.to_csv('evaluationResults.csv')

plt.show()
