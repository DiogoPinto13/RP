from utils import np, stats
from preProcess import preProcessDataset

def ksTest():
  dfNormalized, dfLabels = preProcessDataset()

  X = dfNormalized.to_numpy()
  classes = dfLabels.to_numpy().flatten()

  isReliable = np.where(classes == 1)
  isNotReliable = np.where(classes == 0)
  featureNames = dfNormalized.columns
 
  Hs = {}
  for i in range(np.shape(X)[1]):
    st = stats.kruskal(
      X[isReliable, i].flatten(), 
      X[isNotReliable, i].flatten()
    )

    # get H for each feature
    Hs[featureNames[i]] = st.statistic
  
  # sort features according to H value
  Hs = sorted(Hs.items(), key=lambda x: x[1],reverse=True)  

  print("Ranked features")
  for feature in Hs:
    print(f"{feature[0]} --> {str(feature[1])}")

ksTest()
