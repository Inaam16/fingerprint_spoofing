import numpy as np
import scipy
import scipy.special
from Utilities import *
from pre_processing import project_PCA, PCA


def mean_and_covariance(Data):
    """Given Data, return its mean and its covariance Matrix"""
    mu = mCol(Data.mean(1))
    DataCentered = Data - mu
    covariance = np.dot(DataCentered, DataCentered.T) / Data.shape[1]
    return mu, covariance


# Compute Z-score normalization (center data and normalize variance)
def Z_score(D):
    return (D - D.mean(1).reshape((D.shape[0], 1))) / (np.var(D, axis = 1).reshape((D.shape[0], 1)) ** 0.5)


# Apply Z-score normalization to test data using mean and variance of training dataset 
def Z_score_eval(DTR, DTE):
    return (DTE - DTR.mean(1).reshape((DTR.shape[0], 1))) / (np.var(DTR, axis = 1).reshape((DTR.shape[0], 1)) ** 0.5)


def PCA_preproccessor(DTR, LTR, DTE, LTE, dim):
    DTR_p, vects, _ = PCA(DTR, dim, components=True)
    DTE_p = project_PCA(DTE, vects)
    return DTR_p, LTR, DTE_p, LTE

# def compute_mean_cov_classes(DTR, LTR):
#     n_classes = np.unique(LTR).size
#     means, covariances = list(), list()
#     for i in range(n_classes):
#         mu, cov = muCov(DTR[:, LTR == i])
#         means.append(mu)
#         covariances.append(cov)
#     return means, covariances














