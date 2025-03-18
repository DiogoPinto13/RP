import utils

def preProcessDataset():
    df = utils.pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
    
    #keep only numeric features
    df = df.select_dtypes(include=['number'])
    
    # nullCounts = df.isna().sum()
    # print("Null values in each column:")
    # print(nullCounts)

    scaler = utils.MinMaxScaler()
    scaler.fit(df)
    dfNormalized = utils.pd.DataFrame(scaler.fit_transform(df))

    print(dfNormalized.head(5))

    return dfNormalized
