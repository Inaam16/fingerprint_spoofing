
import numpy as np
from Metrics import act_DCF, optimal_Bayes_decisions, min_DCF, confusion_matrix, Bayes_risk
import logistic_regression as lr
from Utilities import load
import GMM
# Perform cross validation to evaluate score calibration (scores are 
# calibrated with a linear logistic regression model)
def k_fold_calibration(D, L, k, pi, Cfp, Cfn, pi_T, l, seed=0, just_cal = False):

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

    llr_cal = np.zeros([D.shape[1], ])
    opt_th_decisions = np.zeros([D.shape[1]])

    for count in range(k):

        if start_index + elements > D.shape[1]:
            end_index = D.shape[1]
        else:
            end_index = start_index + elements 

        # Define training and test partitions
        idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
    
        DTE = D[:, idxTest]
        LTE = L[idxTest]

        # Train a logistic regression model for score calibration
        llr_cal[idxTest], _ = lr.linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
        
        if not just_cal: # do not repeat optimal threshold estimation every time
            # Estimate optimal threshold on training set and perform decisions on test set
            _, opt_t = min_DCF(DTR.reshape([DTR.shape[1],]), pi, Cfn, Cfp, LTR)
            opt_th_decisions[idxTest] = 1 * (DTE.reshape([DTE.shape[1],]) > opt_t)

        start_index += elements

    # Subtract theoretical threshold to achieve calibrated scores
    llr_cal -= np.log(pi / (1 - pi))
    # Calculate act DCF for calibrated scores
    actDCF_cal = act_DCF(llr_cal, pi, Cfn, Cfp, L)

    if just_cal:
        actDCF_estimated = 0
    else:
        # Calculate act DCF for optimal estimated threshold
        M = confusion_matrix(L, opt_th_decisions, 2)
        _, actDCF_estimated = Bayes_risk(M, pi, Cfn, Cfp)

    return actDCF_cal, actDCF_estimated


# Perform cross validation to evaluate score calibration techniques and print results
def analyse_scores_kfold(llr, pi, Cfn, Cfp, L, k, pi_T, name):
    
    actDCF = act_DCF(llr, pi, Cfn, Cfp, L)
    # Choose the best value for lambda for logistic regression (try different ones)
    min_actDCF_cal = 1
    best_lambda = 0
    actDCF_estimated = 0
    for l in [0, 1e-6, 1e-3, 0.1, 1]:
        if l == 1: # last iteration, calculate also optimal estimated threshold
            actDCF_cal, actDCF_estimated = k_fold_calibration(llr.reshape([1,llr.shape[0]]), L, k, pi, Cfp, Cfn, pi_T, l, just_cal=False)
        else: # not the last iteration, just evaluate score calibration
            actDCF_cal, actDCF_estimated = k_fold_calibration(llr.reshape([1,llr.shape[0]]), L, k, pi, Cfp, Cfn, pi_T, l, just_cal=True)

        if actDCF_cal < min_actDCF_cal:
            min_actDCF_cal = actDCF_cal
            best_lambda = l

    print("\n\n******* "+name+" *********\n")
    print("act DCF: "+str(actDCF))
    print("act DCF, calibrated scores (logistic regression): "+ str(min_actDCF_cal) + " with best lambda: " + str(best_lambda))
    print("act DCF, estimated threshold: "+ str(actDCF_estimated))



def Bayes_error_plots(llr, true_labels):
    effPriorLogOdds = np.linspace(-3,3,21)
    DCF = np.zeros([effPriorLogOdds.shape[0]])
    minDCF = np.zeros([effPriorLogOdds.shape[0]])

    for index, p_tilde in enumerate(effPriorLogOdds):
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        pred = optimal_Bayes_decisions(llr, pi_tilde, 1, 1)
        M = confusion_matrix(true_labels, pred, 2)
        _, DCF[index] = Bayes_risk(M, pi_tilde, 1, 1)
        minDCF[index], _ = min_DCF(llr, pi_tilde, 1, 1, true_labels)

    return DCF, minDCF


if __name__ == "__main__":

    D, L = load("./Train.txt")
    preprocessor = lr.partial(lr.PCA_preproccessor, dim=6)
    _, llrLR = lr.k_fold_cross_validation(D, L, lr.quadratic_logistic_regression, 5, 1/11, 1, 1, 0, 1/11, preprocessor = preprocessor)
    np.save("llrLR.npy", llrLR)
    llrLR = np.load("./llrLR.npy")
    analyse_scores_kfold(llrLR, 1/11,1, 1, L, 5, 1/11, "LR")



    _, llrGMM = GMM.k_fold_cross_validation_1(D, L,5, 1/11, 1, 1, 8, 2, True, False, True, False, pca_dim=None, seed = 0)
    np.save("llrGMM.npy", llrGMM)
    llrGMM = np.load("./llrGMM.npy")
    analyse_scores_kfold(llrGMM, 1/11, 1, 1, L, 5, 1/11, "GMM")