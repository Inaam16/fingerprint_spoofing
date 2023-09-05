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



# GMM LAB10
# ==== ====
"""
GMM

weighted sum of N gaussians

logpdf_GMM is a function that computes the density of a GMM for a set of samples

gmm = [{w1, mu1, C1}, {w2, mu2, C2}, ...]


"""


import numpy
import scipy.special




Data = numpy.load("C:/Users/inaam/Downloads/GMM_data_4D.npy", allow_pickle= True)
Label = numpy.load("C:/Users/inaam/Downloads/commedia_labels_infpar.npy", allow_pickle=True)


def vcol(vector):
    return vector.reshape((vector.shape[0],1)) 

def vrow(vector):
    return vector.reshape((1,vector.shape[0]))



def logPDF_Gau_ND(X, mu, Covariance): 
    centredX = X-mu
    const = X.shape[0]*numpy.log(2*numpy.pi)
    CovLambda = scipy.linalg.inv(Covariance)
    logDetCov = numpy.linalg.slogdet(Covariance)[1]
    exp = (centredX * numpy.dot(CovLambda, centredX)).sum(0)
    return (-0.5)*(const + logDetCov + exp)

"""


We compute a matrix S
for the cell (1,1):
      => we have log(N(x1|mu1, sigma1)) + log(w1)
for the cell (1,2):
      => we have log(N(x1|mu2, sigma2)) + log(w2)


"""

def logpdf_GMM(X, gmm):
    # S is a matrix of number of rows = number of classes
    # number of columns = number of samples
    S = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(X.shape[1]):
        for(idx, component) in enumerate(gmm):
            S[idx, i] = logPDF_Gau_ND(X[:, i:i+1], component[1], component[2]) + numpy.log(component[0])
    return S, scipy.special.logsumexp(S, axis=0)


#EM algorithm has 2 steps:
#   computing the responsibilities for each component for each sample
#   updating the model parameters  
def EM(X, gmm, psi):
    limit = 1e-6
    loss_new = None
    loss_old = None
    
    while loss_old is None or loss_new - loss_old > limit:
        loss_old = loss_new
        S_j = numpy.zeros((len(gmm), X.shape[1]))
        for idx in range(len(gmm)):
            S_j[idx, :] = logPDF_Gau_ND(X, gmm[idx][1], gmm[idx][2]) + numpy.log(gmm[idx][0])
        S_m = vrow(scipy.special.logsumexp(S_j, axis=0))

        # S_j, S_m = logpdf_GMM(X, gmm)
        S_p = numpy.exp(S_j - S_m)
        loss_new = numpy.mean(S_m)
        # M-Step
        Z = numpy.sum(S_p, axis=1)

        F = numpy.zeros((X.shape[0], len(gmm)))
        for idx in range(len(gmm)):
            F[:, idx] = numpy.sum(S_p[idx, :]*X, axis=1)
        
        S = numpy.zeros((X.shape[0], X.shape[0], len(gmm)))
        for idx in range(len(gmm)):
            S[:, :, idx] = numpy.dot(S_p[idx, :]*X, X.T)
        
        mu_new = F/Z
        C_new = S/Z

        for idx in range(len(gmm)):
            C_new[:, :, idx] -= numpy.dot(vcol(mu_new[:, idx]), vrow(mu_new[:, idx]))


        w_new = Z/numpy.sum(Z)

        gmm_new = [((w_new[idx]), vcol(mu_new[:, idx]), C_new[:, :, idx]) for idx in range(len(gmm))]

        for i in range(len(gmm_new)):
         C_new = gmm_new[i][2]
         u, s, _ = numpy.linalg.svd(C_new)
         s[s < psi] = psi
         gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], numpy.dot(u, vcol(s)*u.T))
        gmm = gmm_new
        #print(loss_new)
    return gmm_new

def LBG(X, gmm, n, alpha, psi):
    

    for i in range(len(gmm_init)):
        C_new = gmm_init[i][2]
        u, s, _ = numpy.linalg.svd(C_new)
        s[s < psi] = psi
        gmm_init[i] = (gmm_init[i][0], gmm_init[i][1], numpy.dot(u, vcol(s)*u.T))
    
    gmm_init = EM(X, gmm, psi)

    for i in range(n):
        print(i)
        print(gmm_init)
        gmm_new = []
        for g in range(len(gmm_init)):
            w_new = gmm_init[g][0]/2
            C_g = gmm_init[g][2]
            u, s, _ = numpy.linalg.svd(C_g)
            d = u[:, 0:1]*s[0]**0.5 * alpha
            gmm_new.append((w_new, gmm_init[g][1] + d, C_g))
            gmm_new.append((w_new, gmm_init[g][1] - d, C_g))  
        gmm_init = EM(X, gmm_new)
    return gmm_init

if __name__ == "__main__":

    gmm_init = [[0.3333333333333333, [[-2.0]], [[1.0]]], [0.3333333333333333, [[0.0]], [[1.0]]], [0.3333333333333333, [[2.0]], [[1.0]]]]
    #gmm_final = EM(Data, gmm_init)
    #print(gmm_final)
    gmm_final = LBG(Data, gmm_init, 1, 0.1)
    print(gmm_final)


# ==== ====
