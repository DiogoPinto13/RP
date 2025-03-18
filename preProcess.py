import utils

def preProcessDataset():
    df = utils.pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
    dfLabels = df["label"]
    df = df.drop('label', axis=1)
    #keep only numeric features
    df = df.select_dtypes(include=['number'])
    
    # nullCounts = df.isna().sum()
    # print("Null values in each column:")
    # print(nullCounts)

    scaler = utils.MinMaxScaler()
    scaler.fit(df)
    dfNormalized = utils.pd.DataFrame(scaler.fit_transform(df))

    print(dfNormalized.head(5))

    return dfNormalized, dfLabels

def featureSelectionPCA(df):

    pca = utils.PCA(n_components=40)
    pca.fit(df)
    dfPCA = utils.pd.Dataframe(pca.transform(df))
    print("Transformed Data (Principal Components):")
    print(dfPCA)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    
    return dfPCA

def featureSelectionLDA(dfData, dfLabels):
    lda = utils.LinearDiscriminantAnalysis(n_components=1)
    lda.fit(dfData, dfLabels)
    dfDataLDA = utils.pd.DataFrame(lda.transform(dfData))
    print("Transformed Data (Linear Discriminants):")
    print(dfDataLDA)
    print("Explained Variance Ratio: ", lda.explained_variance_ratio_)

    return dfDataLDA