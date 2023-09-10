# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:53:17 2023

@author: maria
"""

# def plot_bayess_error(llr, label):
#     effPriorLogOdds = np.linspace(-3, 3, 21)
#     # for each value of p we compute the effective prior
#     plt.figure()
#     llr = llr.ravel()
#     dcf = np.zeros(21)
#     min_dcf = np.zeros(21)

#     for idx, p in enumerate(effPriorLogOdds):
#         pi = 1 / (1 + np.exp(-p))
#         dcf[idx] = metrics.DCF_norm(llr, label, pi, 1, 1)
#         min_dcf[idx] = metrics.minimun_DCF_norm(llr, label, pi, 1, 1)[0]

#     plt.plot(effPriorLogOdds, dcf, label="actDCF", color="b")
#     plt.plot(effPriorLogOdds, min_dcf, label="minDCF", color="r")
#     plt.ylim([0, 1])
#     plt.xlim([-3, 3])
#     plt.show()



# def plot_roc(llr, label):
#     tpr, fpr = metrics.roc_points(llr, label)
#     plt.plot(fpr, tpr)
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.grid()
#     plt.legend()
#     plt.show()



# Logistic Regression
# def wrapLogisticRegression(trainData, trainLabel, l):
#     dim = trainData.shape[0]
#     trainEntropy = trainLabel*2.0-1.0
#     def logisticRegression(vect):
#         W = MLCore.mCol(vect[0:dim])
#         bias = vect[-1]
#         scores = np.dot(W.T, trainData) + bias
#         perSampleLoss = np.logaddexp(0, -trainEntropy * scores)
#         loss = perSampleLoss.mean() + 0.5*l*np.linalg.norm(W)**2
#         return loss
#     return logisticRegression


# def LR(trainData, trainLabel, testData, testLabel, lamb):
#     logReg = wrapLogisticRegression(trainData, trainLabel, lamb[0])
#     x0 = np.zeros(trainData.shape[0]+1) # number of data + bias
#     x0t, f0t, d = scipy.optimize.fmin_l_bfgs_b(logReg, x0 = x0 , approx_grad=True)
#     W, bias = MLCore.mCol(x0t[0:testData.shape[0]]), x0t[-1]
#     posterioProb = np.dot(W.T, testData) + bias
#     predictLabel = (posterioProb > 0)*1
#     return testLabel == predictLabel , W, bias

# def compute_LR_score(DTE, w, b):
#     scores = np.dot(w.T, DTE) + b
#     return scores




# def wrapLogisticRegression(trainData, trainLabel, l):
#     dim = trainData.shape[0]
#     trainEntropy = trainLabel * 2.0 - 1.0

#     def logisticRegression(vect):
#         W = mCol(vect[0:dim])
#         bias = vect[-1]
#         scores = np.dot(W.T, trainData) + bias
#         perSampleLoss = np.logaddexp(0, -trainEntropy * scores)
#         loss = perSampleLoss.mean() + 0.5 * l * np.linalg.norm(W) ** 2
#         return loss

#     return logisticRegression


# def LR(trainData, trainLabel, testData, testLabel, lamb):
#     """Logistic Regression"""
#     logReg = wrapLogisticRegression(trainData, trainLabel, lamb[0])
#     x0 = np.zeros(trainData.shape[0] + 1)  # number of data + bias
#     x0t, f0t, d = scipy.optimize.fmin_l_bfgs_b(logReg, x0=x0, approx_grad=True)
#     W, bias = mCol(x0t[0 : testData.shape[0]]), x0t[-1]
#     posterioProb = np.dot(W.T, testData) + bias
#     predictLabel = (posterioProb > 0) * 1
#     return testLabel == predictLabel













