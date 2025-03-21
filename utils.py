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
        if (featureReductionOption != 3):
            print("Please choose number of features for reduction: ")
            numberFeaturesReduction = int(input("Number of features ->" ))

        print("Please choose an option for classification: ")
        print("[1] - Fisher LDA")
        print("[2] - Euclidian Minimum Distance Classifier")
        print("[3] - Mahalanobis Minimum Distance Classifier")
        classifierOption = int(input("Option -> "))

    print("numberFeaturesSelection", numberFeaturesSelection)
    return (
        optionTrain,
        featureSelectionOption,
        numberFeaturesSelection,
        featureReductionOption,
        numberFeaturesReduction,
        classifierOption
    )
