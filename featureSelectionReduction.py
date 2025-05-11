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

def removeCorrelatedFeatures(selectedFeaturesDict, dfData, featureNames, threshold = 0.9):
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
  
  return {index: score for index, score in selectedFeaturesDict.items() if index not in correlated_features}

def featureSelectionKsTest(dfData, dfLabels, numberFeatures = None, removeCorrelated = True, returnRanked = False):
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

  if removeCorrelated:
    # remove H value for correlated features
    Hs = removeCorrelatedFeatures(Hs, dfData, featureNames)

  # select features with highest H value
  Hs = sorted(Hs.items(), key=lambda item: item[1], reverse=True) 
  Hs = Hs[:numberFeatures]
  selectedFeatures = [feature for feature, _ in Hs]

  if returnRanked:
    # return ranked features for automatic testing
    return Hs
  else:
    # return df with selected features
    dfData = utils.pd.DataFrame(dfData, columns=featureNames)
    return dfData[selectedFeatures] 

def featureSelectionRocCurve(dfData, dfLabels, numberFeatures = None, removeCorrelated=True, returnRanked = False):
  numberFeatures = numberFeatures if numberFeatures is not None else dfData.shape[1]

  featureNames = dfData.columns.values
  dfData = dfData.to_numpy()

  # get AUC score for each feature
  aucScores = {}
  for i in range(utils.np.shape(dfData)[1]):
    auc = utils.roc_auc_score(dfLabels, dfData[:, i])
    aucScores[featureNames[i]] = auc

  if removeCorrelated:
    # remove AUC value for correlated features
    aucScores = removeCorrelatedFeatures(aucScores, dfData, featureNames)

  # select features with highest AUC value
  aucScores = sorted(aucScores.items(), key=lambda item: item[1], reverse=True) 
  aucScores = aucScores[:numberFeatures]
  selectedFeatures = [feature for feature, _ in aucScores]

  if returnRanked:
    # return ranked features for automatic testing
    return aucScores
  else:
    # return df with selected features
    dfData = utils.pd.DataFrame(dfData, columns=featureNames)
    return dfData[selectedFeatures] 


