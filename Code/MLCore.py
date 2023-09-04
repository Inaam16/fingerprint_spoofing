import numpy
import scipy

# reshape a numpy row into a column
def mCol(row):
    return row.reshape((row.size, 1))
# reshape a numpy column into a row
def mRow(col):
    return col.reshape((1, col.size))

# given Data it returns its mean mu and its covariance Matrix
def muCov(Data):
    mu = mCol(Data.mean(1))
    DataCentered = Data - mu
    Covariance = numpy.dot(DataCentered,DataCentered.T)/Data.shape[1]
    return mu, Covariance

# applies Principal Component Analysis to the Data and returns the #dim most important dimensions
def PCA(Data, dim):
    Covariance = muCov(Data)[1]
    eigVal, eigVect = numpy.linalg.eigh(Covariance)
    dimEigVect=eigVect[:,::-1][:,0:dim]
    return dimEigVect, eigVal

# return a new dataset projected in the new subspace 
# NEW: addded the dimEigVect as returned value 
def PCA_projected(Data, dim):
    dimEigVect, eigVal = PCA(Data, dim)
    DTR = numpy.dot(dimEigVect.T, Data)
    return DTR, dimEigVect, eigVal

#
def SwSb(Data, Label):
    Sw = 0
    Sb = 0
    mu = mCol(Data.mean(1))
    for id in range(Label.max()+1):
        lData =  Data[:,Label == id]
        lmu = mCol(lData.mean(1))
        Sb += lData.shape[1] * numpy.dot(lmu-mu, (lmu-mu).T)
        Sw += numpy.dot(lData-lmu, (lData-lmu).T)
    Sb /= Data.shape[1]
    Sw /= Data.shape[1]
    return Sw, Sb

# applies Linear Discriminant to the Data and returns the #dim most important dimensions
def LDA(Data, Label, dim):
    Sw, Sb = SwSb(Data, Label)
    eigVal, eigVect = scipy.linalg.eigh(Sb, Sw)
    dimeigVect=eigVect[:,::-1][:,0:dim]
    return dimeigVect

def LDA_projected(Data, Label, dim):
    dimeigVect = LDA(Data, Label, dim)
    DTR = numpy.dot(dimeigVect.T, Data)
    return DTR, dimeigVect


#logaritmic Probability Density Function Gaussian-Normal Density
def logPDF_Gau_ND(X, mu, Covariance): 
    centredX = X-mu
    const = X.shape[0]*numpy.log(2*numpy.pi)
    CovLambda = scipy.linalg.inv(Covariance)
    logDetCov = numpy.linalg.slogdet(Covariance)[1]
    exp = (centredX * numpy.dot(CovLambda, centredX)).sum(0)
    return (-0.5)*(const + logDetCov + exp)

def loglikelihood(Data, mu, Covariance):
    return logPDF_Gau_ND(Data, mu, Covariance).sum()

# Data split (deprecated -> see kfold)
def DBSplit(Data, Label, trainPartition = 2,testPartition = 1, seed = 0):
    nTrain = int(Data.shape[1]*trainPartition/(testPartition + trainPartition))
    numpy.random.seed(seed)
    idx = numpy.random.permutation(Data.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    trainData   = Data[:, idxTrain]
    testData    = Data[:, idxTest]
    trainLabel  = Label[idxTrain]
    testLabel   = Label[idxTest]

    return (trainData, trainLabel), (testData, testLabel)

# apply the main process of thr Multivariate Gaussian Classifier
# trainMuCov is a list containing  (mu, Covariance) for each class
# priorProb is a list containing the prior Probability for each class
def applyMVG(trainMuCov, testData, priorProb):
    logScore = []
    logPriorProb = [numpy.log(prob) for prob in priorProb]
    for id, prob in enumerate(priorProb):
        logScore.append(logPDF_Gau_ND(testData, trainMuCov[id][0], trainMuCov[id][1])+logPriorProb[id])
    logJointScore = numpy.vstack(logScore)
    logMarginalScore = mRow(scipy.special.logsumexp(logJointScore, axis=0))
    logPpostProb = logJointScore-logMarginalScore
    predictLabel = numpy.argmax(logPpostProb, axis = 0) # index of the max value
    return predictLabel

# Calculate the Multivariate Gaussian Classifier through train Data and applies it
# to the test Data returning the Data correctly classified
def MVG(trainData, trainLabel, testData, testLabel, priorProb):
    labeledMuCov = []
    for id in range(trainLabel.max()+1):
        lData = trainData[:,trainLabel==id]
        lmu, lCovariance = muCov(lData)
        labeledMuCov.append((lmu,lCovariance))
    predictLabel = applyMVG(labeledMuCov,testData, priorProb)
    return testLabel == predictLabel

# Calculate the naive Bias Gaussian Classifier through train Data and applies it
# to the test Data returning the Data correctly classified
def naiveBayesGC(trainData, trainLabel, testData, testLabel, priorProb):
    labeledMuCov = []
    for id in range(trainLabel.max()+1):
        lData = trainData[:,trainLabel==id]
        lmu, lCovariance = muCov(lData)
        lCovariance = lCovariance*numpy.eye(lCovariance.shape[0])
        labeledMuCov.append((lmu,lCovariance))
    predictLabel = applyMVG(labeledMuCov,testData, priorProb)
    return testLabel == predictLabel

# Calculate the tied Multivariate Gaussian Classifier through train Data and applies it
# to the test Data returning the Data correctly classified
def tied_MVG(trainData, trainLabel, testData, testLabel, priorProb):
    Covariance = numpy.zeros((trainData.shape[0], trainData.shape[0]))
    lMu = []
    for id in range(trainLabel.max()+1):
        lData = trainData[:,trainLabel==id]
        tmu, tCovariance = muCov(lData)
        lMu.append(tmu)
        Covariance += tCovariance*lData.shape[1]
    Covariance /= trainData.shape[1]
    labeledMuCov =[(mu, Covariance) for mu in lMu]
    predictLabel = applyMVG(labeledMuCov,testData, priorProb)
    return testLabel == predictLabel

# Calculate the naive tied Bias Gaussian Classifier through train Data and applies it
# to the test Data returning the Data correctly classified
def tied_naiveBayesGC(trainData, trainLabel, testData, testLabel, priorProb):
    Covariance = numpy.zeros((trainData.shape[0], trainData.shape[0]))
    lMu = []
    for id in range(trainLabel.max()+1):
        lData = trainData[:,trainLabel==id]
        tmu, tCovariance = muCov(lData)
        lMu.append(tmu)
        Covariance += tCovariance*lData.shape[1]
    Covariance /= trainData.shape[1]
    Covariance = Covariance*numpy.eye(Covariance.shape[0])
    labeledMuCov =[(mu, Covariance) for mu in lMu]
    predictLabel = applyMVG(labeledMuCov,testData, priorProb)
    return testLabel == predictLabel

# divide the data into k partition and using the "leave one out" test the classifier (function)
def k_fold(Data, Label, priorProb, function, k = 5, seed = 0):
    foldSize = Data.shape[1]//k
    accuracy = 0
    #prior per class Probability
    priorPCProb = priorProb if isinstance(priorProb, list) else [priorProb for i in range(Label.max()+1)]
    numpy.random.seed(seed)
    idx = numpy.random.permutation(Data.shape[1])
    for i in range(k):
        foldIdx = (i*foldSize, (i+1)*foldSize)
        testIdx = idx[foldIdx[0]:foldIdx[1]]
        trainIdx = numpy.concatenate([idx[:foldIdx[0]], idx[foldIdx[1]:]])
        trainData, trainLabel = Data[:, trainIdx], Label[trainIdx]
        testData, testLabel = Data[:, testIdx], Label[testIdx]
        accuracy += function(trainData, trainLabel, testData, testLabel, priorPCProb).mean()
    return accuracy/k
 
def wrapLogisticRegression(trainData, trainLabel, l):
    dim = trainData.shape[0]
    trainEntropy = trainLabel*2.0-1.0
    def logisticRegression(vect):
        W = mCol(vect[0:dim])
        bias = vect[-1]
        scores = numpy.dot(W.T, trainData) + bias
        perSampleLoss = numpy.logaddexp(0, -trainEntropy * scores)
        loss = perSampleLoss.mean() + 0.5*l*numpy.linalg.norm(W)**2
        return loss
    return logisticRegression

#logistic Regression
def LR(trainData, trainLabel, testData, testLabel, lamb):
    logReg = wrapLogisticRegression(trainData, trainLabel, lamb[0])
    x0 = numpy.zeros(trainData.shape[0]+1) # number of data + bias
    x0t, f0t, d = scipy.optimize.fmin_l_bfgs_b(logReg, x0 = x0 , approx_grad=True)
    W, bias = mCol(x0t[0:testData.shape[0]]), x0t[-1]
    posterioProb = numpy.dot(W.T, testData) + bias
    predictLabel = (posterioProb > 0)*1
    return testLabel == predictLabel