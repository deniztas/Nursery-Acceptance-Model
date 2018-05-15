import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as getTime
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from copy import deepcopy
from IPython import embed

lineCount = 80
columnNames = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'target']
skipPlots = True
randSeed = 666 ## TO DO: Change randomly
testPercent = 0.2
kFoldingValue = 5

print(" ==== Step 1 - Load Dataset ==== ")
dataset = pd.read_csv('dataset/nursery.csv', sep=',', header=None)
dataset.dropna(axis=0, how='any')
for i in range(len(columnNames)):
    dataset[i] = dataset[i].astype('category') # Defining all types to categorical
dataset.columns = columnNames # Changes names of columns
print("Dataset is loaded.")
print("-"*lineCount)

print(" ==== Step 2 - Summarize Dataset ==== ")
nDatarow = dataset.shape[0]
nColumn = dataset.shape[1]-1
print("Number of dataset instance:\t" + str(nDatarow))
print("Number of descriptive features:\t" + str(nColumn))
print("Number of target feature(s):\t1")
print("-"*lineCount)
print("First 5 row:")
print(dataset.head(5))
print("-"*lineCount)
print("Statistics for categorical features:")
print(dataset.describe(include='category'))
print("-"*lineCount)
print("Target count of dataset:")
print(dataset.groupby('target').size())
print("-"*lineCount)
for colName in ['target']: # dataset.columns
    dataset[colName].value_counts().plot(kind='bar', title=str(colName))
    if not skipPlots:
        plt.show()

print(" ==== Step 3 - Creating Dummies ==== ")
print(str(len(dataset.columns)) + " categorical feature found.")
dataset = pd.get_dummies(dataset, columns=list(dataset.columns))
print(str(len(dataset.columns)) + " dummy feature created.")
print("-"*lineCount)

print(" ==== Step 4 - Seperating Dataset ==== ")
trainDataset, testDataset = model_selection.train_test_split(dataset, test_size=testPercent, random_state=randSeed)
print("-"*lineCount)

print(" ==== Step 5 - Seperating Columns ==== ")
regexContains = 'target'
regexNotContains = '^((?!target).)*$' # https://stackoverflow.com/questions/406230/regular-expression-to-match-a-line-that-doesnt-contain-a-word#answer-406408
X_train = trainDataset.filter(regex=regexNotContains)
y_train = trainDataset.filter(regex=regexContains)
X_test = testDataset.filter(regex=regexNotContains)
y_test = testDataset.filter(regex=regexContains)
print("-"*lineCount)

print(" ==== Step 6 - Creating Machine Learning Models ==== ")
models = [
    ('Decision Tree', DecisionTreeClassifier(criterion='gini')),
    ('Decision Tree', DecisionTreeClassifier(criterion='entropy')),
    #('Random Forest', RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=1)),
    #('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', n_jobs=1)),
    #('MLP one-layer', MLPClassifier(hidden_layer_sizes=(16 ), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=100)),
    #('MLP tri-layer', MLPClassifier(hidden_layer_sizes=(16, 32, 16 ), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=100)),
]
print("Number of models going to be run:" + str(len(models)))
print("Models:")
for modelName, _ in models:
    print("\t" + str(modelName))
print("-"*lineCount)

print(" ==== Step 7 - Training ==== ")
for modelname, modelObj in models:
    print(modelname + " training has started")
    start = getTime()
    kfold = model_selection.KFold(n_splits=kFoldingValue, random_state=randSeed, shuffle=True)
    totAccr = 0
    maxAccr = (0, None)
    for trainIndices, evaluateIndices in kfold.split(X_train):
        X_training = X_train.iloc[trainIndices]
        y_training = y_train.iloc[trainIndices]
        X_evaluating = X_train.iloc[evaluateIndices]
        y_evaluating = y_train.iloc[evaluateIndices]

        modelObj.fit(X_training, y_training)
        y_predicted = modelObj.predict(X_evaluating)
        y_predicted = pd.DataFrame(data=y_predicted, columns=y_evaluating.columns)
        accr = accuracy_score(y_evaluating.idxmax(axis=1), y_predicted.idxmax(axis=1))
        if accr > maxAccr[0]:
            maxAccr = (accr, deepcopy(trainIndices))
        totAccr += accr

    if maxAccr[0] > 0:
        X_training = X_train.iloc[maxAccr[1]]
        y_training = y_train.iloc[maxAccr[1]]
        modelObj.fit(X_training, y_training)
    print("\tAverage Accuracy: " + str(totAccr / kFoldingValue))
    print(modelname + " training has finished in " + str(getTime() - start) + " seconds")
    print("-"*lineCount)

print(" ==== Step 8 - Test Predictions ==== ")
print("Confusion matrix labels in order: ")
print(" ".join(list(map(str, y_test.columns))))
print("-"*lineCount)
for modelname, modelObj in models:
    y_predicted = modelObj.predict(X_test)
    y_predicted = pd.DataFrame(data=y_predicted, columns=y_test.columns)
    accr = accuracy_score(y_test.idxmax(axis=1), y_predicted.idxmax(axis=1))
    recl = recall_score(y_test.idxmax(axis=1), y_predicted.idxmax(axis=1), average='macro')
    prec = precision_score(y_test.idxmax(axis=1), y_predicted.idxmax(axis=1), average='macro')
    confMatrix = confusion_matrix(y_test.idxmax(axis=1), y_predicted.idxmax(axis=1), labels=y_evaluating.columns)
    print(modelName)
    print("\tAverage Accuracy: " + str(accr))
    print("\tConfusion Matrix: ")
    for confRow in confMatrix:
        print("\t" + "\t".join(list(map(str, confRow.tolist()))))
    print("\tRecall: " + str(recl))
    print("\tPrecision: " + str(prec))
    print("-"*lineCount)

