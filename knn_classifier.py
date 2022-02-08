#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:34:14 2021

@author: juliann
"""

import pandas as pd
import math
from statistics import mode
from heapq import nsmallest
from sklearn import metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def knn(df, k):
    ''' func: step_two
    parameters: df, k
    returns: accuracy, precision, recall, f1'''
    results = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    predicted = []
    actual = []
    # cycles through every row of the dataframe
    for i in range(len(df.index)):
        # creates a dataframe of every OTHER row as training data
        trainers = df.drop([i], axis=0)
        # finds the distances between the test coords and all the training coords
        neighbors = distances(i, df.loc[i], trainers)
        result = []
        # finds the k lowest distances
        for x in k_nearest(neighbors, k):
            # checking the target value for each of the k nearest neighbors
            result.append(df.loc[x]['D'])
        # checks if the majority of the k lowest distances are equal to y_test, whether or not our classifier was correct
        results.append(correct_checker(result, df.loc[i]['D']))
        # adding the predicted and actual value for the target attribute 'D' to lists to generate metrics info from sci-kit learn
        actual.append(df.loc[i]['D'])
        predicted.append(mode(result))
        # finds out if our classifier returned a false positive, false negative, true positive, or true negative (building a confusion matrix)
        tn = confusion_builder(result, df.loc[i]['D'], tn, tp, fn, fp)[0]
        tp = confusion_builder(result, df.loc[i]['D'], tn, tp, fn, fp)[1]
        fn = confusion_builder(result, df.loc[i]['D'], tn, tp, fn, fp)[2]
        fp = confusion_builder(result, df.loc[i]['D'], tn, tp, fn, fp)[3]  
    # generating a confusion matrix heatmap for the results using the current k value
    confusion_visualizer(k, tp, fp, fn, tn)
    # returns score(accuracy), k, precision, recall, f1 score
    return score(results), precision_recall_f1(actual, predicted)[0], precision_recall_f1(actual, predicted)[1], precision_recall_f1(actual, predicted)[2]

    
    
def distances(j, test, trainers):
    ''' func: distances
    parameters: i (index of test row), test (one row of the df), trainers (all the other rows in a new df)
    returns: dict of the distances between the test point and the training points, with indeces being instance indicators and values being euclidian distnace'''
    x = 'X1'
    y = 'X4'
    distances = {}
    for i in range(len(trainers.index)):
        if i != j:
            distances[i] = euclidian([test[x], test[y]], [trainers.iloc[i][x], trainers.iloc[i][y]])
    return distances


def euclidian(a,b):
    ''' function: euclidian
    parameters: two lists of equal value
    returns: the euclidian distance between two points'''
    d = 0
    for i in range(len(a)):
        d += (a[i] - b[i]) ** 2
    return math.sqrt(d)
    

def k_nearest(distances, k):
    ''' func: k_nearest
    parameters: distances(dict), k
    returns: list of keys of the k smallest values in the given dict'''
    return nsmallest(k, distances, key = distances.get)
    

def correct_checker (results, target):
    ''' function: correct checker
    parameters: results (list of k nearest training targets), test target(int), ints indicating true and false positives and negatives
    return: boolean '''
    return mode(results) == target

def confusion_builder(results, target, tn, tp, fn, fp):
    '''function: confusion_builder
    parameters: results (list of k nearest training targets), test target(int)
    returns: int'''
    if mode(results) == 0 and target == 0:
        tn += 1
    if mode(results) == 1 and target == 1:
        tp += 1
    if mode(results) == 0 and target == 1:
        fn += 1
    if mode(results) == 1 and target == 0:
        fp += 1
    return tn, tp, fn, fp
    
    
def score(results):
    ''' func: score
    parameter: results (list)
    returns: float '''
    correct = 0
    total = len(results)
    for i in results:
        if i == True:
            correct += 1
    return (correct / total)

def confusion_visualizer(k, tp, fp, fn, tn):
    ''' func: confusion_visualizer
    parameters: four ints representing the elements of a confusion matrix
    returns: seaborn heatmap'''
    square = [(tp, fp), (fn, tn)]
    matrix = pd.DataFrame(square, columns = ['Positive', 'Negative'], index = ['Positive', 'Negative'])
    plt.figure()
    visual = sns.heatmap(matrix, annot = True)
    plt.savefig(str(k) + '_nearest_neighbors_confusion_matrix.png')
    

def precision_recall_f1(real, pred):
    ''' func: precision_recall_f1
    parameters: real, predicted (arrays)
    returns: precision, recall, f1 (floats) '''
    pred = np.array(pred)
    real = np.array(real)
    p = metrics.precision_score(real, pred)
    r = metrics.recall_score(real, pred)
    f1 = metrics.f1_score(real, pred)
    return p, r, f1
    
    
def overall_visualizer(k, a, p, r, f1):
    ''' func: overall_visualizer
    paramters: k, a, p, r, f1 (lists)
    returns: graphic '''
    plt.figure()
    plt.plot(k, a, label = "accuracy")
    plt.plot(k, p, label = 'precision')
    plt.plot(k, r, label = "recall")
    plt.plot(k, f1, label = "f1 score")
    plt.xlabel('k')
    plt.title('the effects of a different k on the metrics of a homegrown knn classifier')
    plt.legend()
    plt.savefig('overall graph.png')
    

def main():
    
    # step 1
    df = pd.read_csv('knn_data.csv', encoding = 'utf-16')
    ks = []
    a = []
    p = []
    r = []
    f1 = []
    # step 2, finding the knn accuacy for different values of k
    for k in range(1, 33, 2):
        print('k: ', str(k))
        print('accuracy: ' + str(knn(df,k)[0]))
        # step 3, finding precision, recall, and f1 score
        print('precision: ' + str(knn(df,k)[1]))
        print('recall: ' + str(knn(df,k)[2]))
        print('f1 score: ' + str(knn(df,k)[3]))
        # step 4, creating a graphic displaying the effects of a different k on the metrics
        ks.append(k)
        a.append(knn(df,k)[0])
        p.append(knn(df,k)[1])
        r.append(knn(df,k)[2])
        f1.append(knn(df,k)[3])
    overall_visualizer(ks, a, p, r, f1)


if __name__ == '__main__':
    main()

    



    












    
    





    





    
    
       
      


    
   
       
       
       
      
        
    
