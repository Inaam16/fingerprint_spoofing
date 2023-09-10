import numpy as np


def confusion_matrix(true_labels, predicted_labels, n_labels):
    """Compute confusion matrix with given predictions"""
    conf_matrix = np.zeros([n_labels, n_labels])

    for current_true_label in range(n_labels):
        for current_predicted_label in range(n_labels):
            conf_matrix[current_predicted_label, current_true_label] = sum(
                predicted_labels[true_labels == current_true_label] == current_predicted_label
            )

    return conf_matrix


def Bayes_risk(M, pi, Cfn, Cfp):
    FNR = M[0, 1] / (M[0, 1] + M[1, 1])
    FPR = M[1, 0] / (M[0, 0] + M[1, 0])

    DCFu = pi * Cfn * FNR + (1 - pi) * Cfp * FPR
    B_dummy = min(pi * Cfn, (1 - pi) * Cfp)
    DCF = DCFu / B_dummy

    return DCFu, DCF


# Calculate min DCF by trying all possible thresholds
def min_DCF(llr, pi, Cfn, Cfp, true_labels):
    possible_t = np.concatenate(
        (np.array([min(llr) - 0.1]), (np.unique(llr)), np.array([max(llr) + 0.1]))
    )

    minDCF = 10
    opt_t = 0

    for t in possible_t:
        PredictedLabels = np.zeros([llr.shape[0]])
        PredictedLabels[llr > t] = 1
        M = confusion_matrix(true_labels, PredictedLabels, 2)
        _, DCF = Bayes_risk(M, pi, Cfn, Cfp)
        if DCF < minDCF:
            minDCF = DCF
            opt_t = t

    return minDCF, opt_t


