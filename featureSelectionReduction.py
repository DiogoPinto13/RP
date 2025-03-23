import utils

def featureReductionPCA(dfData, dfLabels, numberFeatures = None, criterionPCAOption = None):
  numberFeatures = numberFeatures if numberFeatures is not None else dfData.shape[1]

  pca = utils.PCA(n_components=dfData.shape[1])
  pca.fit(dfData)

  # select features for reduction
  components = numberFeatures
  eigenvalues = pca.explained_variance_
  if (criterionPCAOption == 1):
    # kaiser criterion
    kaiserComponents = sum(eigenvalues > 1)
    components = kaiserComponents
  elif (criterionPCAOption == 2):
    # scree test
    diff = utils.np.diff(eigenvalues) * -1 # get difference between eigenvalues
    threshold = 0.1 * max(diff) # set threshold for diff
    screeComponents = utils.np.argmax(diff < threshold) + 1
    components = screeComponents
  else:
    pca = utils.PCA(n_components=components)
  
  dfPCA = utils.pd.DataFrame(pca.fit_transform(dfData))
  return dfPCA

def featureReductionLDA(dfData, dfLabels, numberFeatures = None, criterionPCAOption = None):
  # n_components is min(n_features, n_classes - 1)
  # for our problem, n_classes = 2, so: n_components=1
  lda = utils.LinearDiscriminantAnalysis(n_components=1) 

  lda.fit(dfData, dfLabels)
  dfDataLDA = utils.pd.DataFrame(lda.transform(dfData))
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



