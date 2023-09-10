import numpy as np
import scipy
import scipy.special
from Utilities import *


def mean_and_covariance(Data):
    """Given Data, return its mean and its covariance Matrix"""
    mu = mCol(Data.mean(1))
    DataCentered = Data - mu
    covariance = np.dot(DataCentered, DataCentered.T) / Data.shape[1]
    return mu, covariance


def compute_mean_cov_classes(DTR, LTR):
    n_classes = np.unique(LTR).size
    means, covariances = list(), list()
    for i in range(n_classes):
        mu, cov = muCov(DTR[:, LTR == i])
        means.append(mu)
        covariances.append(cov)
    return means, covariances


def compute_eff_prior(prior, Cfn, Cfp):
    """Comupte the effective prior"""
    return (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)


def principal_components(Data, dim):
    """Apply Principal Component Analysis to the Data and return the dim most important dimensions"""
    mu, cov = mean_and_covariance(Data)
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    dim_eigen_vects = eigen_vectors[:, ::-1][:, 0:dim]
    dim_eigen_vals = eigen_values[::-1][0:dim]
    return dim_eigen_vects, dim_eigen_vals


def project_PCA(Data, eigen_vectors):
    DTR = np.dot(eigen_vectors.T, Data)
    return DTR


def PCA(Data, dim, *, components=False):
    """Return a new dataset projected in the new subspace
    NEW: addded the dimEigVect as returned value"""
    eigen_vectors, eigen_values = principal_components(Data, dim)
    DTR = np.dot(eigen_vectors.T, Data)
    if components:
        return DTR, eigen_vectors, eigen_values
    return DTR


def SwSb(Data, Label):
    Sw = 0
    Sb = 0
    mu = mCol(Data.mean(1))
    for id in range(Label.max() + 1):
        lData = Data[:, Label == id]
        lmu = mCol(lData.mean(1))
        Sb += lData.shape[1] * np.dot(lmu - mu, (lmu - mu).T)
        Sw += np.dot(lData - lmu, (lData - lmu).T)
    Sb /= Data.shape[1]
    Sw /= Data.shape[1]
    return Sw, Sb


def LDA(Data, Label, dim):
    """Apply Linear Discriminant to the Data and returns the #dim most important dimensions"""
    Sw, Sb = SwSb(Data, Label)
    eigVal, eigVect = scipy.linalg.eigh(Sb, Sw)
    dimeigVect = eigVect[:, ::-1][:, 0:dim]
    return dimeigVect


def LDA_projected(Data, Label, dim):
    dimeigVect = LDA(Data, Label, dim)
    DTR = np.dot(dimeigVect.T, Data)
    return DTR, dimeigVect


def logpdf_1sample(x, mu, C):
    """Compute the log density of a single feature vector x"""
    P = np.linalg.inv(C)
    res = -0.5 * x.shape[0] * np.log(2 * np.pi)
    res += -0.5 * np.linalg.slogdet(C)[1]
    res += -0.5 * np.dot((x - mu).T, np.dot(P, (x - mu))).ravel()
    return res


def logpdf_GAU_ND(X, mu, C):
    """Logaritmic Probability Density Function Gaussian-Normal Density
    Returns density of samples belonging to a given class having mu and C"""
    Y = [logpdf_1sample(X[:, i : i + 1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()


# def logPDF_Gau_ND(X, mu, Covariance):
#     """Logaritmic Probability Density Function Gaussian-Normal Density
#     Returns density of samples belonging to a given class having mu and C"""
#     centredX = X - mu
#     const = X.shape[0] * np.log(2 * np.pi)
#     CovLambda = scipy.linalg.inv(Covariance)
#     _, logDetCov = np.linalg.slogdet(Covariance)
#     exp = (centredX * np.dot(CovLambda, centredX)).sum(0)
#     return (-0.5) * (const + logDetCov + exp)

# def log_GAU_ND(X, mean, covariance):
#         m = X.shape[0]
#         _, logdet = np.linalg.slogdet(covariance)
#         inv = np.linalg.inv(covariance)
#         dif = X - mean
#         # Compute and return log likelihood
#         return -0.5 * (m * np.log(2 * np.pi) + logdet + (dif.T @ inv @ dif).diagonal())


def loglikelihood(Data, mu, Covariance):
    return logPDF_Gau_ND(Data, mu, Covariance).sum()


def naiveBayesGC(trainData, trainLabel, testData, testLabel, priorProb):
    """Calculate the naive Bias Gaussian Classifier through train Data and applies it
    to the test Data returning the Data correctly classified"""
    labeledMuCov = []
    for id in range(trainLabel.max() + 1):
        lData = trainData[:, trainLabel == id]
        lmu, lCovariance = muCov(lData)
        lCovariance = lCovariance * np.eye(lCovariance.shape[0])
        labeledMuCov.append((lmu, lCovariance))
    predictLabel = applyMVG(labeledMuCov, testData, priorProb)
    return testLabel == predictLabel


def tied_MVG(trainData, trainLabel, testData, testLabel, priorProb):
    """Calculate the tied Multivariate Gaussian Classifier through train Data and applies it
    to the test Data returning the Data correctly classified"""
    Covariance = np.zeros((trainData.shape[0], trainData.shape[0]))
    lMu = []
    for id in range(trainLabel.max() + 1):
        lData = trainData[:, trainLabel == id]
        tmu, tCovariance = muCov(lData)
        lMu.append(tmu)
        Covariance += tCovariance * lData.shape[1]
    Covariance /= trainData.shape[1]
    labeledMuCov = [(mu, Covariance) for mu in lMu]
    predictLabel = applyMVG(labeledMuCov, testData, priorProb)
    return testLabel == predictLabel


def tied_naiveBayesGC(trainData, trainLabel, testData, testLabel, priorProb):
    """Calculate the naive tied Bias Gaussian Classifier through train Data and applies it
    to the test Data returning the Data correctly classified"""
    Covariance = np.zeros((trainData.shape[0], trainData.shape[0]))
    lMu = []
    for id in range(trainLabel.max() + 1):
        lData = trainData[:, trainLabel == id]
        tmu, tCovariance = muCov(lData)
        lMu.append(tmu)
        Covariance += tCovariance * lData.shape[1]
    Covariance /= trainData.shape[1]
    Covariance = Covariance * np.eye(Covariance.shape[0])
    labeledMuCov = [(mu, Covariance) for mu in lMu]
    predictLabel = applyMVG(labeledMuCov, testData, priorProb)
    return testLabel == predictLabel


def k_fold(Data, Label, priorProb, function, k=5, seed=0):
    """Divide the data into k partitions and using the "leave one out" test the classifier (function)"""
    foldSize = Data.shape[1] // k
    remainder = Data.shape[1] % k
    accuracy = 0

    # Handle prior probability
    unique_labels = np.unique(Label)
    if isinstance(priorProb, list):
        priorPCProb = priorProb
    else:
        priorPCProb = [priorProb for _ in unique_labels]

    np.random.seed(seed)
    idx = np.random.permutation(Data.shape[1])

    for i in range(k):
        if i < remainder:
            start_idx = i * (foldSize + 1)
            end_idx = start_idx + foldSize + 1
        else:
            start_idx = i * foldSize + remainder
            end_idx = start_idx + foldSize

        testIdx = idx[start_idx:end_idx]
        trainIdx = np.concatenate([idx[:start_idx], idx[end_idx:]])

        trainData, trainLabel = Data[:, trainIdx], Label[trainIdx]
        testData, testLabel = Data[:, testIdx], Label[testIdx]

        accuracy += function(trainData, trainLabel, testData, testLabel, priorPCProb).mean()

    return accuracy / k


def wrapLogisticRegression(trainData, trainLabel, l):
    dim = trainData.shape[0]
    trainEntropy = trainLabel * 2.0 - 1.0

    def logisticRegression(vect):
        W = mCol(vect[0:dim])
        bias = vect[-1]
        scores = np.dot(W.T, trainData) + bias
        perSampleLoss = np.logaddexp(0, -trainEntropy * scores)
        loss = perSampleLoss.mean() + 0.5 * l * np.linalg.norm(W) ** 2
        return loss

    return logisticRegression


def LR(trainData, trainLabel, testData, testLabel, lamb):
    """Logistic Regression"""
    logReg = wrapLogisticRegression(trainData, trainLabel, lamb[0])
    x0 = np.zeros(trainData.shape[0] + 1)  # number of data + bias
    x0t, f0t, d = scipy.optimize.fmin_l_bfgs_b(logReg, x0=x0, approx_grad=True)
    W, bias = mCol(x0t[0 : testData.shape[0]]), x0t[-1]
    posterioProb = np.dot(W.T, testData) + bias
    predictLabel = (posterioProb > 0) * 1
    return testLabel == predictLabel


# GMM LAB10
# ==== ====
"""
GMM

weighted sum of N gaussians

logpdf_GMM is a function that computes the density of a GMM for a set of samples

gmm = [{w1, mu1, C1}, {w2, mu2, C2}, ...]


"""
# Data = np.load("C:/Users/inaam/Downloads/GMM_data_4D.npy", allow_pickle=True)
# Label = np.load(
#     "C:/Users/inaam/Downloads/commedia_labels_infpar.npy", allow_pickle=True
# )


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
    S = np.zeros((len(gmm), X.shape[1]))
    for i in range(X.shape[1]):
        for idx, component in enumerate(gmm):
            S[idx, i] = logPDF_Gau_ND(X[:, i : i + 1], component[1], component[2]) + np.log(
                component[0]
            )
    return S, scipy.special.logsumexp(S, axis=0)


def EM(X, gmm, psi):
    """EM algorithm has 2 steps:
    - computing the responsibilities for each component for each sample
    - updating the model parameters"""
    limit = 1e-6
    loss_new = None
    loss_old = None

    while loss_old is None or loss_new - loss_old > limit:
        loss_old = loss_new
        S_j = np.zeros((len(gmm), X.shape[1]))
        for idx in range(len(gmm)):
            S_j[idx, :] = logPDF_Gau_ND(X, gmm[idx][1], gmm[idx][2]) + np.log(gmm[idx][0])
        S_m = vrow(scipy.special.logsumexp(S_j, axis=0))

        # S_j, S_m = logpdf_GMM(X, gmm)
        S_p = np.exp(S_j - S_m)
        loss_new = np.mean(S_m)
        # M-Step
        Z = np.sum(S_p, axis=1)

        F = np.zeros((X.shape[0], len(gmm)))
        for idx in range(len(gmm)):
            F[:, idx] = np.sum(S_p[idx, :] * X, axis=1)

        S = np.zeros((X.shape[0], X.shape[0], len(gmm)))
        for idx in range(len(gmm)):
            S[:, :, idx] = np.dot(S_p[idx, :] * X, X.T)

        mu_new = F / Z
        C_new = S / Z

        for idx in range(len(gmm)):
            C_new[:, :, idx] -= np.dot(vcol(mu_new[:, idx]), vrow(mu_new[:, idx]))

        w_new = Z / np.sum(Z)

        gmm_new = [
            ((w_new[idx]), vcol(mu_new[:, idx]), C_new[:, :, idx]) for idx in range(len(gmm))
        ]

        for i in range(len(gmm_new)):
            C_new = gmm_new[i][2]
            u, s, _ = np.linalg.svd(C_new)
            s[s < psi] = psi
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], np.dot(u, vcol(s) * u.T))
        gmm = gmm_new
        # print(loss_new)
    return gmm_new


def LBG(X, gmm, n, alpha, psi):
    for i in range(len(gmm_init)):
        C_new = gmm_init[i][2]
        u, s, _ = np.linalg.svd(C_new)
        s[s < psi] = psi
        gmm_init[i] = (gmm_init[i][0], gmm_init[i][1], np.dot(u, vcol(s) * u.T))

    gmm_init = EM(X, gmm, psi)

    for i in range(n):
        print(i)
        print(gmm_init)
        gmm_new = []
        for g in range(len(gmm_init)):
            w_new = gmm_init[g][0] / 2
            C_g = gmm_init[g][2]
            u, s, _ = np.linalg.svd(C_g)
            d = u[:, 0:1] * s[0] ** 0.5 * alpha
            gmm_new.append((w_new, gmm_init[g][1] + d, C_g))
            gmm_new.append((w_new, gmm_init[g][1] - d, C_g))
        gmm_init = EM(X, gmm_new)
    return gmm_init


if __name__ == "__main__":
    gmm_init = [
        [0.3333333333333333, [[-2.0]], [[1.0]]],
        [0.3333333333333333, [[0.0]], [[1.0]]],
        [0.3333333333333333, [[2.0]], [[1.0]]],
    ]
    # gmm_final = EM(Data, gmm_init)
    # print(gmm_final)
    gmm_final = LBG(Data, gmm_init, 1, 0.1)
    print(gmm_final)
