import utils
import classifiers

def logStatsMsg(logFile, msg, reset=False):
  print(msg)
  utils.Path(logFile).parent.mkdir(parents=True, exist_ok=True)
  mode = "w" if reset else "a"
  with open(logFile, mode) as f:
    f.write(msg + "\n")

def checkCombinationsNormality(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = utils.os.path.join(basePath, "statisticalTestResult.txt")

  logStatsMsg(
    logFile,
    f"1. Shapiro-Wilk normality test results for classifier '{classifier.__name__}' and test '{testType}':\n",
    True
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )
  
  allParametric = True
  for i, combDir in enumerate(combinationDirs, start=1):
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()

    _, p = utils.test_normal_sw(combinationValues)
    logStatsMsg(logFile, f"Combination {i}: p = {p:.4f} → {'Normal' if p > 0.05 else 'Not normal'}")

    if p <= 0.05:
      allParametric = False

  logStatsMsg(
    logFile,
    "\nResult: All combinations are parametric." if allParametric else "\nResult: At least one combination is not parametric."
  )
  return allParametric

def checkNonParametricCombinationsDifference(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = basePath / "statisticalTestResult.txt"

  logStatsMsg(
    logFile,
    f"\n2. Due to the normality test indicating non-parametric data, performing Kruskal-Wallis test for differences between parameter combinations of classifier '{classifier.__name__}' and test '{testType}':"
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )

  allCombinationValues = []
  for combDir in combinationDirs:
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()
    allCombinationValues.append(combinationValues)

  _, p = utils.kruskal_wallis(allCombinationValues)

  logStatsMsg(logFile, f"Result: p = {p:.4f}")
  logStatsMsg(logFile, "Conclusion: No significant difference between combinations." if p > 0.05 else "Conclusion: At least one combination differs significantly from the others.")
  return p <= 0.05

def checkParametricCombinationsDifference(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = basePath / "statisticalTestResult.txt"

  logStatsMsg(
    logFile,
    f"\n2. Data is parametric. Performing One-Way ANOVA to test for differences between parameter combinations of classifier '{classifier.__name__}' and test '{testType}':"
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )

  allCombinationValues = []
  for combDir in combinationDirs:
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()
    allCombinationValues.append(combinationValues)

  _, p = utils.one_way_ind_anova(allCombinationValues)

  logStatsMsg(logFile, f"Result: p = {p:.4f}")
  if p > 0.05:
    logStatsMsg(logFile, "Conclusion: No significant difference between combinations.")
  else:
    logStatsMsg(logFile, "Conclusion: At least one combination differs significantly from the others.")

  return p <= 0.05

def checkPairwiseParametricComparisons(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = basePath / "statisticalTestResult.txt"

  logStatsMsg(
    logFile,
    f"\n3. Pairwise Independent t-tests for combinations of classifier '{classifier.__name__}' and test '{testType}':"
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )

  allCombinationValues = []
  for combDir in combinationDirs:
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()
    allCombinationValues.append(combinationValues)

  n = len(allCombinationValues)
  wins_uncorrected = [0] * n
  wins_corrected = [0] * n

  pair_indices = []
  raw_p_values = []
  t_values = []
  means = []

  for (i, data1), (j, data2) in utils.combinations(enumerate(allCombinationValues), 2):
    t, p = utils.t_test_ind(data1, data2)
    pair_indices.append((i, j))
    raw_p_values.append(p)
    t_values.append(t)
    means.append((utils.np.mean(data1), utils.np.mean(data2)))

  logStatsMsg(logFile, "\n→ Results without Bonferroni correction:")
  for (i, j), p, t, (mean1, mean2) in zip(pair_indices, raw_p_values, t_values, means):
    if p <= 0.05:
      if mean1 > mean2:
        wins_uncorrected[i] += 1
        result = f"Combination {i+1} beats Combination {j+1} (raw p = {p:.4f}, t = {t:.2f})"
      else:
        wins_uncorrected[j] += 1
        result = f"Combination {j+1} beats Combination {i+1} (raw p = {p:.4f}, t = {t:.2f})"
    else:
      result = f"Combination {i+1} vs Combination {j+1}: Not significant (raw p = {p:.4f}, t = {t:.2f})"
    logStatsMsg(logFile, result)

  corrected_results = utils.multipletests(raw_p_values, alpha=0.05, method='bonferroni')
  adjusted_p_values = corrected_results[1]

  logStatsMsg(logFile, "\n→ Results with Bonferroni correction:")
  for (i, j), adj_p, t, (mean1, mean2) in zip(pair_indices, adjusted_p_values, t_values, means):
    if adj_p <= 0.05:
      if mean1 > mean2:
        wins_corrected[i] += 1
        result = f"Combination {i+1} beats Combination {j+1} (adjusted p = {adj_p:.4f}, t = {t:.2f})"
      else:
        wins_corrected[j] += 1
        result = f"Combination {j+1} beats Combination {i+1} (adjusted p = {adj_p:.4f}, t = {t:.2f})"
    else:
      result = f"Combination {i+1} vs Combination {j+1}: Not significant (adjusted p = {adj_p:.4f}, t = {t:.2f})"
    logStatsMsg(logFile, result)

  ranked_uncorrected = sorted(enumerate(wins_uncorrected), key=lambda x: x[1], reverse=True)
  logStatsMsg(logFile, "\n→ Ranking based on number of pairwise wins (without correction):")
  for rank, (idx, winCount) in enumerate(ranked_uncorrected, start=1):
    logStatsMsg(logFile, f"{rank}. Combination {idx+1} — {winCount} wins")

  ranked_corrected = sorted(enumerate(wins_corrected), key=lambda x: x[1], reverse=True)
  logStatsMsg(logFile, "\n→ Ranking based on number of pairwise wins (with Bonferroni correction):")
  for rank, (idx, winCount) in enumerate(ranked_corrected, start=1):
    logStatsMsg(logFile, f"{rank}. Combination {idx+1} — {winCount} wins")

def checkPairwiseNonParametricComparisons(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = basePath / "statisticalTestResult.txt"

  logStatsMsg(
    logFile,
    f"\n3. Pairwise Mann-Whitney U tests for combinations of classifier '{classifier.__name__}' and test '{testType}':"
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )

  allCombinationValues = []
  for combDir in combinationDirs:
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()
    allCombinationValues.append(combinationValues)

  n = len(allCombinationValues)
  wins_uncorrected = [0] * n
  wins_corrected = [0] * n

  pair_indices = []
  raw_p_values = []
  means = []

  for (i, data1), (j, data2) in utils.combinations(enumerate(allCombinationValues), 2):
    _, p = utils.mann_whitney(data1, data2)
    pair_indices.append((i, j))
    raw_p_values.append(p)
    means.append((utils.np.mean(data1), utils.np.mean(data2)))

  logStatsMsg(logFile, "\n→ Results without Bonferroni correction:")
  for (i, j), p, (mean1, mean2) in zip(pair_indices, raw_p_values, means):
    if p <= 0.05:
      if mean1 > mean2:
        wins_uncorrected[i] += 1
        result = f"Combination {i+1} beats Combination {j+1} (raw p = {p:.4f})"
      else:
        wins_uncorrected[j] += 1
        result = f"Combination {j+1} beats Combination {i+1} (raw p = {p:.4f})"
    else:
      result = f"Combination {i+1} vs Combination {j+1}: Not significant (raw p = {p:.4f})"
    logStatsMsg(logFile, result)

  corrected_results = utils.multipletests(raw_p_values, alpha=0.05, method='bonferroni')
  adjusted_p_values = corrected_results[1]

  logStatsMsg(logFile, "\n→ Results with Bonferroni correction:")
  for (i, j), adj_p, (mean1, mean2) in zip(pair_indices, adjusted_p_values, means):
    if adj_p <= 0.05:
      if mean1 > mean2:
        wins_corrected[i] += 1
        result = f"Combination {i+1} beats Combination {j+1} (adjusted p = {adj_p:.4f})"
      else:
        wins_corrected[j] += 1
        result = f"Combination {j+1} beats Combination {i+1} (adjusted p = {adj_p:.4f})"
    else:
      result = f"Combination {i+1} vs Combination {j+1}: Not significant (adjusted p = {adj_p:.4f})"
    logStatsMsg(logFile, result)

  ranked_uncorrected = sorted(enumerate(wins_uncorrected), key=lambda x: x[1], reverse=True)
  logStatsMsg(logFile, "\n→ Ranking based on number of pairwise wins (without correction):")
  for rank, (idx, winCount) in enumerate(ranked_uncorrected, start=1):
    logStatsMsg(logFile, f"{rank}. Combination {idx+1} — {winCount} wins")

  ranked_corrected = sorted(enumerate(wins_corrected), key=lambda x: x[1], reverse=True)
  logStatsMsg(logFile, "\n→ Ranking based on number of pairwise wins (with Bonferroni correction):")
  for rank, (idx, winCount) in enumerate(ranked_corrected, start=1):
    logStatsMsg(logFile, f"{rank}. Combination {idx+1} — {winCount} wins")

def baseStatTest(classifier, testType, metric="fScore", onlyTwo=False):
  isParametric = checkCombinationsNormality(classifier, testType, metric)

  if onlyTwo:
    if isParametric: checkPairwiseParametricComparisons(classifier, testType, metric)
    else: checkPairwiseNonParametricComparisons(classifier, testType, metric) 
  else:
    if not isParametric:
      isDifferent = checkNonParametricCombinationsDifference(classifier, testType, metric) 
      if isDifferent:
        checkPairwiseNonParametricComparisons(classifier, testType, metric)
    else:
      isDifferent = checkParametricCombinationsDifference(classifier, testType, metric)
      if isDifferent:
        checkPairwiseParametricComparisons(classifier, testType, metric)

def comparisonClassifiers():
  pass

baseStatTest(comparisonClassifiers, "comparisonClassifiers", "fScore")