import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy import stats


def showMenu():
    print("Please choose an option for feature reduction / selection: ")
    print("[1] - PCA")
    print("[2] - LDA")
    print("[3] - KS")
    featureReductionSelection = int(input("Option -> "))

    print("Please choose an option for classification: ")
    print("[1] - Fisher LDA")
    print("[2] - Euclidian Minimum Distance Classifier")
    print("[3] - Mahalanobis Minimum Distance Classifier")
    classifierSelection = int(input("Option -> "))

    return featureReductionSelection, classifierSelection
