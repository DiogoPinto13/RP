import utils

def featureReductionPCA(dfData, dfLabels, numberFeatures = None):
  numberFeatures = numberFeatures if numberFeatures is not None else dfData.shape[1]

  pca = utils.PCA(n_components=numberFeatures)
  pca.fit(dfData)
  dfPCA = utils.pd.DataFrame(pca.transform(dfData))
  print("Transformed Data (Principal Components):")
  print(dfPCA)
  print("Explained Variance Ratio:", pca.explained_variance_ratio_)
  
  return dfPCA

def featureReductionLDA(dfData, dfLabels, numberFeatures = None):
  lda = utils.LinearDiscriminantAnalysis(n_components=1)
  lda.fit(dfData, dfLabels)
  dfDataLDA = utils.pd.DataFrame(lda.transform(dfData))
  print("Transformed Data (Linear Discriminants):")
  print(dfDataLDA)
  print("Explained Variance Ratio: ", lda.explained_variance_ratio_)

  return dfDataLDA

def featureSelectionKsTest(dfData, dfLabels, numberFeatures = None):
  numberFeatures = numberFeatures if numberFeatures is not None else dfData.shape[1]

  featureNames = dfData.columns.values
  dfData = dfData.to_numpy()
  classes = dfLabels.to_numpy().flatten()

  isReliable = utils.np.where(classes == 1)
  isNotReliable = utils.np.where(classes == 0)
 
  Hs = {}
  for i in range(utils.np.shape(dfData)[1]):
    st = utils.stats.kruskal(
      dfData[isReliable, i].flatten(), 
      dfData[isNotReliable, i].flatten()
    )

    # get H for each feature
    Hs[featureNames[i]] = st.statistic

  # get correlation matrix
  correlationMatrix = utils.np.corrcoef(dfData, rowvar=False)

  # get features with correlation bigger than 90%
  threshold = 0.9
  correlated_features = set()
  numberOfFeatures = len(featureNames)
  for i in range(numberOfFeatures):
    for j in range(i + 1, numberOfFeatures):  
      if abs(correlationMatrix[i, j]) > threshold:
        correlated_features.add(j)

  # remove H value for correlated features
  Hs = {index: score for index, score in Hs.items() if index not in correlated_features}

  # select features with highest H value
  Hs = sorted(Hs.items(), key=lambda item: item[1], reverse=True) 
  Hs = Hs[:numberFeatures]
  selected_features = [feature for feature, _ in Hs]

  # return df with selected features
  dfData = utils.pd.DataFrame(dfData, columns=featureNames)
  return dfData[selected_features]



