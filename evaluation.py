import utils

def main(
  numberFeaturesSelection,
  featureSelection,
  numberFeaturesReduction,
  featureReduction,
  classifier,
  dfTargetTest,
  dfPredictions,
  generateCSV=True
):
  # get confusion matrix
  tn, fp, fn, tp = utils.confusion_matrix(dfTargetTest, dfPredictions).ravel()

  # get metrics
  accuracy = utils.accuracy_score(dfTargetTest, dfPredictions)
  precision = utils.precision_score(dfTargetTest, dfPredictions)
  recall = utils.recall_score(dfTargetTest, dfPredictions)
  fScore = utils.f1_score(dfTargetTest, dfPredictions)

  # format result df
  dfResult = utils.pd.DataFrame([{
    "featureSelection": featureSelection,
    "numberFeaturesSelection": numberFeaturesSelection,
    "featureReduction": featureReduction,
    "numberFeaturesReduction": numberFeaturesReduction,
    "classifier": classifier,
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "fScore": fScore
  }])

  if generateCSV:
    # save result df as csv
    timestamp = utils.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/output_{timestamp}.csv"
    dfResult.to_csv(path, index=False)

  return dfResult
