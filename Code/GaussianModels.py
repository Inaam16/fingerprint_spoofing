import numpy as np
from MLCore import mRow, mean_and_covariance, PCA, project_PCA
from Utilities import load
import constants as cnst
from Metrics import *


# def _applyMVG(trainMuCov, testData, priorProb):
#     """Apply the main process of the Multivariate Gaussian Classifier
#     trainMuCov is a list containing  (mu, Covariance) for each class
#     priorProb is a list containing the prior Probability for each class"""
#     logScore = []
#     logPriorProb = [np.log(prob) for prob in priorProb]
#     for i, prob in enumerate(priorProb):
#         logScore.append(
#             logPDF_Gau_ND(testData, trainMuCov[i][0], trainMuCov[i][1]) + logPriorProb[i]
#         )
#     logJointScore = np.vstack(logScore)
#     logMarginalScore = mRow(scipy.special.logsumexp(logJointScore, axis=0))
#     logPpostProb = logJointScore - logMarginalScore
#     predictLabel = np.argmax(logPpostProb, axis=0)  # index of the max value
#     return predictLabel


# def MVG(trainData, trainLabel, testData, testLabel, priorProb):
#     """Calculate the Multivariate Gaussian Classifier through train Data and applies it
#     to the test Data returning the Data correctly classified"""
#     labeledMuCov = []
#     for i in range(trainLabel.max() + 1):
#         lData = trainData[:, trainLabel == i]
#         lmu, lCovariance = mean_and_covariance(lData)
#         labeledMuCov.append((lmu, lCovariance))
#     predictLabel = _applyMVG(labeledMuCov, testData, priorProb)
#     return testLabel == predictLabel


def GAU_logpdf_ND(x, mu, C):
    M = x.shape[0]
    _, logSigma = np.linalg.slogdet(C)

    if x.shape[1] == 1:
        logN = (
            -M / 2 * np.log(2 * np.pi)
            - 0.5 * logSigma
            - 0.5 * np.dot(np.dot((x - mu).T, numpy.linalg.inv(C)), (x - mu))
        )
    else:
        logN = (
            -M / 2 * np.log(2 * np.pi)
            - 0.5 * logSigma
            - 0.5 * np.diagonal(np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x - mu)))
        )

    return logN


# Compute llr from class conditional log probabilities (just subtract the
# values for the two classes)
def compute_llr(s):
    if s.shape[0] != 2:
        return 0
    return s[1, :] - s[0, :]


def MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn):
    mu = []
    sigma = []
    n_class = len(cnst.CLASS_NAMES)
    # Compute mean and covariance for each class
    for c in range(n_class):
        m, s = mean_and_covariance(DTR[:, LTR == c])
        mu.append(m)
        sigma.append(s)

    # Compute class-conditional log probabilities for each class
    S = np.zeros([n_class, DTE.shape[1]])
    for c in range(n_class):
        # log domain
        S[c, :] = GAU_logpdf_ND(DTE, mu[c], sigma[c])

    # Compute log-likelihood ratios and minDCF
    llr = compute_llr(S)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF


# Perform k-fold cross validation on test data for the specified model
def k_fold_cross_validation(D, L, classifier, k, pi, Cfp, Cfn, preprocessor=None, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

    llr = np.zeros(
        [
            D.shape[1],
        ]
    )

    for count in range(k):
        # Define training and test partitions
        if start_index + elements > D.shape[1]:
            end_index = D.shape[1]
        else:
            end_index = start_index + elements

        idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]

        DTE = D[:, idxTest]
        LTE = L[idxTest]

        if preprocessor:
            DTR, LTR, DTE, LTE = preprocessor(DTR, LTR, DTE, LTE)

        # Train the classifier and compute llr on the current partition
        llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, pi, Cfp, Cfn)
        start_index += elements

    # Evaluate results after all k-fold iterations (when all llr are available)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF


def PCA_preproccessor(DTR, LTR, DTE, LTE, dim):
    DTR_p, vects, _ = PCA(DTR, dim, components=True)
    DTE_p = project_PCA(DTE, vects)
    return DTR_p, LTR, DTE_p, LTE


# Train a naive Bayes Gaussian classifier and evaluate it on test data
def naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn):
    n_class = len(cnst.CLASS_NAMES)
    mu = []
    sigma = []
    # Compute mean and covariance for each class
    for c in range(n_class):
        m, s = mean_and_covariance(DTR[:, LTR == c])
        mu.append(m)
        # Keep only the diagonal of the covariance matrix (naive Bayes approach)
        sigma.append(s * np.eye(s.shape[0]))

    S = np.zeros([n_class, DTE.shape[1]])
    for c in range(n_class):
        # log domain
        S[c, :] = GAU_logpdf_ND(DTE, mu[c], sigma[c])

    # Compute log-likelihood ratios and minDCF
    llr = compute_llr(S)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF


if __name__ == "__main__":
    from functools import partial

    D, L = load("../Train.txt")
    DTE, LTE = load("../Test.txt")

    score = []
    for i in range(1, 11):
        preprocessor = partial(PCA_preproccessor, dim=i)
        score.append(
            k_fold_cross_validation(
                D, L, naive_Bayes, 5, 0.5, 10, 1, seed=0, preprocessor=preprocessor
            )
        )
        print(i, score[-1])

    with open("../results_naive_bayes", "w") as outfile:
        for i in range(1, 11):
            outfile.write(f"{i}, {score[i-1]}\n")
    import matplotlib.pyplot as plt

    plt.plot(range(1, 11), score)
    plt.show()
