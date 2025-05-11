import numpy as np
from datetime import datetime
import os
from numpy.linalg import inv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import scipy.stats as st
from pathlib import Path
from itertools import combinations
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

def getClassifierLabel(classifierFunctionName):
    classifiers = {
        "fisherLDA": "Fisher LDA",
        "eucludeanMinimumDistanceClassifier": "Euclidean MDC",
        "mahalanobisMinimumDistanceClassifier": "Mahalanobis MDC",
        "svmClassifier" : "SVM classifier",
        "KNNClassifier": "KNN classifier",
        "naiveBayesClassifier": "Naive Bayes classifier"
    }
    return classifiers[classifierFunctionName]

def getReductionLabel(classifierFunctionName):
    reductions = {
        "featureReductionPCA": "PCA Reduction"
    }
    return reductions[classifierFunctionName]

def showMenu():
    print("Please choose an option:")
    print("[1] - automatic testing for curve")
    print("[2] - automatic testing for 30 iterations")
    print("[3] - normal")
    print("[4] - automatic fine tuning")
    print("[5] - automatic feature selection Roc Curve results")
    print("[6] - automatic feature selection Kruskal-Wallis results")
    print("[7] - automatic feature correlation results")
    optionTrain = int(input("Option -> "))
    numberFeaturesSelection = None
    numberFeaturesReduction = None
    featureSelectionOption = None
    featureReductionOption = None
    classifierOption = None
    criterionPCAOption = None
    removeCorrelated = None
    
    if optionTrain == 3:
        print("Please choose an option for feature selection: ")
        print("[1] - KS")
        print("[2] - ROC Curve")
        print("[3] - None")
        featureSelectionOption = int(input("Option -> "))
        if (featureSelectionOption != 3):
            print("Please choose number of features for selection: ")
            numberFeaturesSelection = int(input("Number of features (Total = 50) -> "))
            print("Do you want to remove correlated features? [y/n]")
            removeCorrelated = input("-> ").strip().lower() == "y"

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
        print("[4] - SVM Classifier")
        print("[5] - KNN Classifier")
        print("[6] - Naive Bayes Classifier")
        classifierOption = int(input("Option -> "))

    return (
        optionTrain,
        featureSelectionOption,
        numberFeaturesSelection,
        featureReductionOption,
        numberFeaturesReduction,
        removeCorrelated,
        classifierOption,
        criterionPCAOption
    )

def test_normal_sw(data):
    """Shapiro-Wilk"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.shapiro(norm_data)

def kruskal_wallis(data):
    """
    non parametric
    many samples
    independent
    """     
    H,pval = st.kruskal(*data)
    return (H,pval)

def mann_whitney(data1, data2):
    """
    non parametric
    two samples
    independent
    """    
    return st.mannwhitneyu(data1, data2)

def one_way_ind_anova(data):
    """
    parametric
    many samples
    independent
    """
    F,pval = st.f_oneway(*data)
    return (F,pval)

def t_test_ind(data1,data2, eq_var=True):
    """
    parametric
    two samples
    independent
    """
    t,pval = st.ttest_ind(data1,data2, equal_var=eq_var)
    return (t,pval)