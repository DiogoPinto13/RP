import numpy as np
from datetime import datetime
import os
from numpy.linalg import inv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt

def getClassifierLabel(classifierFunctionName):
    classifiers = {
        "fisherLDA": "Fisher LDA",
        "eucludeanMinimumDistanceClassifier": "Euclidean MDC",
        "mahalanobisMinimumDistanceClassifier": "Mahalanobis MDC"
    }
    return classifiers[classifierFunctionName]

def getReductionLabel(classifierFunctionName):
    reductions = {
        "featureReductionPCA": "PCA Reduction",
        "featureReductionLDA": "LDA Reduction"
    }
    return reductions[classifierFunctionName]

def showMenu():
    print("Please choose an option:")
    print("[1] - automatic testing for curve")
    print("[2] - automatic testing for 30 iterations")
    print("[3] - normal")
    optionTrain = int(input("Option -> "))
    numberFeaturesSelection = None
    numberFeaturesReduction = None
    featureSelectionOption = None
    featureReductionOption = None
    classifierOption = None
    criterionPCAOption = None
    
    if optionTrain == 3:
        print("Please choose an option for feature selection: ")
        print("[1] - KS")
        print("[2] - None")
        featureSelectionOption = int(input("Option -> "))
        if (featureSelectionOption != 2):
            print("Please choose number of features for selection: ")
            numberFeaturesSelection = int(input("Number of features -> "))

        print("Please choose an option for feature reduction: ")
        print("[1] - PCA")
        print("[2] - LDA")
        print("[3] - None")
        featureReductionOption = int(input("Option -> "))
        if (featureReductionOption == 1):
            print("Please select how you want to select features for reduction: ")
            print("[-1] - Use a criterion (Kaiser Criterion or Scree Test)")
            print("[N] - Number of features")            
            numberFeaturesReduction = int(input("Option -> " ))

            if (numberFeaturesReduction == -1):
                print("[1] - Kaiser Criterion")
                print("[2] - Scree Test")
                criterionPCAOption = int(input("Number of features -> "))

        print("Please choose an option for classification: ")
        print("[1] - Fisher LDA")
        print("[2] - Euclidean Minimum Distance Classifier")
        print("[3] - Mahalanobis Minimum Distance Classifier")
        classifierOption = int(input("Option -> "))

    return (
        optionTrain,
        featureSelectionOption,
        numberFeaturesSelection,
        featureReductionOption,
        numberFeaturesReduction,
        classifierOption,
        criterionPCAOption
    )
