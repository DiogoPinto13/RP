import utils

def fisherLDA(args):
	dfTrain = args["dfTrain"]
	dfTest = args["dfTest"]
	dfTargetTrain = args["dfTargetTrain"]
	dfTargetTest = args["dfTargetTest"]
	lda = utils.LinearDiscriminantAnalysis()
	lda.fit(dfTrain, dfTargetTrain)

	dfPredictions = lda.predict(dfTest)

	return dfTargetTest, dfPredictions 

def eucludeanMinimumDistanceClassifier(args):
	dfTrain = args["dfTrain"]
	dfTest = args["dfTest"]
	dfTargetTrain = args["dfTargetTrain"]
	dfTargetTest = args["dfTargetTest"]
	# get means for each class
	means = []
	sortedClasses = utils.np.sort(dfTargetTrain.unique())
	for classLabel in sortedClasses:
		classSamples = dfTrain[dfTargetTrain == classLabel]
		mean = classSamples.mean(axis=0).values
		means.append(mean)
	means = utils.np.array(means)
	
	# set g function for each class
	gFunctions = []
	for mean in means:
		def gFunction(x, mean=mean): 
			return mean.T @ x - 0.5 * mean @ mean.T
		
		gFunctions.append(gFunction)
	
	# set d function for two first classes
	def d(x):
		# if d >= 0 then isNotReliable else isReliable
		return gFunctions[0](x) - gFunctions[1](x)

	# apply d function to dfTest and get predictions
	dfResults = dfTest.apply(d, axis=1)
	dfPredictions = dfResults.apply(lambda result: 0 if result >= 0 else 1)

	return dfTargetTest, dfPredictions 

def mahalanobisMinimumDistanceClassifier(args):
	dfTrain = args["dfTrain"]
	dfTest = args["dfTest"]
	dfTargetTrain = args["dfTargetTrain"]
	dfTargetTest = args["dfTargetTest"]
	# get means for each class
	means = []
	sortedClasses = utils.np.sort(dfTargetTrain.unique())
	for class_label in sortedClasses:
		classSamples = dfTrain[dfTargetTrain == class_label]
		mean = classSamples.mean(axis=0).values
		means.append(mean)
	means = utils.np.array(means)

	# get covariance matrix for each class
	covarianceMatrices = []
	for classLabel in sortedClasses:
		classSamples = dfTrain[dfTargetTrain == classLabel]
		covarianceMatrix = utils.np.cov(classSamples, rowvar=False)
		covarianceMatrices.append(covarianceMatrix)
	
	# get inversed pooled covariance matrix
	pooledCovariance = sum(covarianceMatrices) / len(sortedClasses)
	if (dfTrain.shape[1] == 1): # dfTrain with just one feature
		inversedPooledCovariance = utils.np.array([[1 / pooledCovariance]])
	else:
		inversedPooledCovariance = utils.inv(pooledCovariance)
	
	# set g function for each class
	gFunctions = []
	for mean in means:
		def gFunction(x, mean=mean): 
			return mean.T @ inversedPooledCovariance @ x - 0.5 * mean.T @ inversedPooledCovariance @ mean
		
		gFunctions.append(gFunction)
	
	# set d function for two first classes
	def d(x):
		# if d > 0 then isNotReliable else isReliable
		return gFunctions[0](x) - gFunctions[1](x)

	# apply d function to dfTest and get predictions
	dfResults = dfTest.apply(d, axis=1)
	dfPredictions = dfResults.apply(lambda result: 0 if result >= 0 else 1)

	return dfTargetTest, dfPredictions 

def svmClassifier(args):
	dfTrain = args["dfTrain"]
	dfTest = args["dfTest"]
	dfTargetTrain = args["dfTargetTrain"]
	dfTargetTest = args["dfTargetTest"]
	c = args["c"]
	gamma = args["gamma"]

	svm = utils.SVC(kernel='rbf', C=c, gamma=gamma,random_state=42)
	svm.fit(dfTrain, dfTargetTrain)

	dfPredictions = svm.predict(dfTest)

	return dfTargetTest, dfPredictions 

def KNNClassifier(args):
	dfTrain = args["dfTrain"]
	dfTest = args["dfTest"]
	dfTargetTrain = args["dfTargetTrain"]
	dfTargetTest = args["dfTargetTest"]
	k = args["k"]

	knn = utils.KNeighborsClassifier(n_neighbors=k)
	knn.fit(dfTrain, dfTargetTrain)

	dfPredictions = knn.predict(dfTest)
	
	return dfTargetTest, dfPredictions