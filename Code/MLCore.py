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


# def compute_mean_cov_classes(DTR, LTR):
#     n_classes = np.unique(LTR).size
#     means, covariances = list(), list()
#     for i in range(n_classes):
#         mu, cov = muCov(DTR[:, LTR == i])
#         means.append(mu)
#         covariances.append(cov)
#     return means, covariances














