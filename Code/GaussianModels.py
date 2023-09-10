import numpy as np
from MLCore import *
from pre_processing import  PCA, project_PCA
from Utilities import load
import constants as cnst
from Metrics import *
from functools import partial
import matplotlib.pyplot as plt

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


# Train a tied Multivariate Gaussian classifier and evaluate it on test data
def tied_MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn):

    mu = []
    tied_sigma = np.zeros([DTR.shape[0], DTR.shape[0]])

    # Compute mean and covariance for each class
    for c in range(n_class):
        m, s = mean_and_covariance_matrix(DTR[:, LTR == c])
        mu.append(m)
        # Compute the tied covariance matrix by averaging all covariance matrixes
        tied_sigma += sum(LTR == c) / DTR.shape[1] * s

    # Compute class-conditional log probabilities for each class    
    S = np.zeros([n_class, DTE.shape[1]])
    for c in range(n_class):
        # log domain
        S[c,:] = GAU_logpdf_ND(DTE, mu[c], tied_sigma)

    # Compute log-likelihood ratios and minDCF
    llr = compute_llr(S)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF




# Train a tied diagonal Gaussian classifier and evaluate it on test data
def tied_naive_MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn):

    mu = []
    tied_sigma = np.zeros([DTR.shape[0], DTR.shape[0]])

    # Compute mean and covariance for each class
    for c in range(n_class):
        m, s = mean_and_covariance_matrix(DTR[:, LTR == c])
        mu.append(m)
        tied_sigma += sum(LTR == c) / DTR.shape[1] * ( s * np.eye(s.shape[0])) 
    
    # Compute class-conditional log probabilities for each class
    S = np.zeros([n_class, DTE.shape[1]])
    for c in range(n_class):
        # log domain
        S[c,:] = GAU_logpdf_ND(DTE, mu[c], tied_sigma)

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







if __name__ == "__main__":
    D, L = load("../Train.txt")

    score = []
    for i in range(1, 11):
        preprocessor = partial(PCA_preproccessor, dim=i)
        score.append(
            k_fold_cross_validation(
                D, L, naive_Bayes, 5, 0.5, 10, 1, seed=0, preprocessor=preprocessor
            )
        )
        print(i, score[-1])

    with open("../Results/Gaussian/results_naive_bayes", "w") as outfile:
        for i in range(1, 11):
            outfile.write(f"{i}, {score[i-1]}\n")
 

    plt.plot(range(1, 11), score)
    plt.show()
