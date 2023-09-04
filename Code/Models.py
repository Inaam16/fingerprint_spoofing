import numpy
import scipy

import MLCore as MLCore



# Gaussian

# === We need to compute the mu and the Covariance matrix for each class ===
def muCov(Data):
    mu = MLCore.mCol(Data.mean(1))
    DataCentered = Data - mu
    Covariance = numpy.dot(DataCentered,DataCentered.T)/Data.shape[1]
    return mu, Covariance

# train_mvg
def compute_mean_cov_classes(DTR, LTR) -> list:
    n_classes = numpy.unique(LTR).size
    means = [muCov(DTR[:, LTR == i])[0] for i in range(n_classes)]
    cov_matrices = [muCov(DTR[:, LTR == i])[1] for i in range(n_classes)]
    return means, cov_matrices



# Returns density of samples belonging to class having mu and C
# FOR ONE CLASS ONLY
def logPDF_Gau_ND(X, mu, Covariance): 
    centredX = X-mu
    const = X.shape[0]*numpy.log(2*numpy.pi)
    CovLambda = scipy.linalg.inv(Covariance)
    logDetCov = numpy.linalg.slogdet(Covariance)[1]
    exp = (centredX * numpy.dot(CovLambda, centredX)).sum(0)
    density_array = (-0.5)*(const + logDetCov + exp)
    return density_array



# ==== MVG ====

# for each class we compute the log-likelihood of the sample belonging to that class

# THIS FUNCTION RETURNS THE SCORES AND NOT THE LABELS
# has 2 rows each row represent the denisty of sample correpsonding to a certain class
def compuet_gaussian_scores(DTE, means, Covariance_matrices):
    S = numpy.zeros((len(means), DTE.shape[1]))
    for i in range(len(means)):
        S[i:i+1, :] = numpy.exp(logPDF_Gau_ND(DTE, means[i], Covariance_matrices[i]))
    llr = numpy.zeros(S.shape[1])
    for i in range(S.shape[1]):
        llr[i] = numpy.log( S[1, i] / S[0,i])
    return llr

# === ==== ====

# === TIED NAIVE BAYES ===
def train_naiveBayesGC(DTE, means, Covariance_matrices):
    Covariance_matrices = [(Covariance_matrices[i]*numpy.eye(Covariance_matrices[i].shape[0])) for i in range(len(means))]
    llr = compuet_gaussian_scores(DTE, means, Covariance_matrices)
    return llr


def train_tied_naiveBayesGC(DTR, LTR, DTE,  means, Covariance_matrices):
    Covariance_matrices = [Covariance_matrices[i]*numpy.eye(DTR.shape[0]) for i in range(len(means))]
    C_shared  = numpy.zeros((Covariance_matrices[0].shape[0], Covariance_matrices[0].shape[0]))
    for i in range(len(means)):
        C_shared += (LTR == i).sum() * Covariance_matrices[i]
    C_shared = [C_shared / DTR.shape[1] for _ in range(len(means))]
    llr = compuet_gaussian_scores(DTE, means, C_shared)
    return llr


def train_tied_MVG(DTR, LTR, DTE,  means, Covariance_matrices):
    C_shared  = numpy.zeros((Covariance_matrices[0].shape[0], Covariance_matrices[0].shape[0]))
    for i in range(len(means)):
        C_shared += (LTR == i).sum() * Covariance_matrices[i]
    C_shared = [C_shared / DTR.shape[1] for _ in range(len(means))]
    llr = compuet_gaussian_scores(DTE, means, C_shared)
    return llr


# THIS FUNCTION RETURNS THE LABELS
# Given : both classes' mu and covariance and the priors  
def applyMVG(trainMuCov, testData, priorProb):
    logScore = []
    logPriorProb = [numpy.log(prob) for prob in priorProb]
    for id, prob in enumerate(priorProb):
        logScore.append(logPDF_Gau_ND(testData, trainMuCov[id][0], trainMuCov[id][1])+logPriorProb[id])
    logJointScore = numpy.vstack(logScore)
    logMarginalScore = MLCore.mRow(scipy.special.logsumexp(logJointScore, axis=0))
    logPpostProb = logJointScore-logMarginalScore
    predictLabel = numpy.argmax(logPpostProb, axis = 0) # index of the max value
    return predictLabel

# Logistic Regression

#

def wrapLogisticRegression(trainData, trainLabel, l):
    dim = trainData.shape[0]
    trainEntropy = trainLabel*2.0-1.0
    def logisticRegression(vect):
        W = MLCore.mCol(vect[0:dim])
        bias = vect[-1]
        scores = numpy.dot(W.T, trainData) + bias
        perSampleLoss = numpy.logaddexp(0, -trainEntropy * scores)
        loss = perSampleLoss.mean() + 0.5*l*numpy.linalg.norm(W)**2
        return loss
    return logisticRegression


def LR(trainData, trainLabel, testData, testLabel, lamb):
    logReg = wrapLogisticRegression(trainData, trainLabel, lamb[0])
    x0 = numpy.zeros(trainData.shape[0]+1) # number of data + bias
    x0t, f0t, d = scipy.optimize.fmin_l_bfgs_b(logReg, x0 = x0 , approx_grad=True)
    W, bias = MLCore.mCol(x0t[0:testData.shape[0]]), x0t[-1]
    posterioProb = numpy.dot(W.T, testData) + bias
    predictLabel = (posterioProb > 0)*1
    return testLabel == predictLabel , W, bias

def compute_LR_score(DTE, w, b):
    scores = numpy.dot(w.T, DTE) + b
    return scores

#





