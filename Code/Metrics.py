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


# ---------------------------------------------------------------------------->
# compute the conf matrix given the predictions and the true labels
def compute_conf_matrix(predictLabel, Label):
    # UNUSED
    conf_matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            conf_matrix[i, j] = (1 * np.bitwise_and(predictLabel == i, Label == j)).sum()
    return conf_matrix


# comapre the llrs with threshold and return the confusion matrix
def compare_with_thre(llr, label, pi, Cfn, Cfp):
    t = -np.log(pi * Cfn / (1 - pi) * Cfp)
    predictions = (llr > t) * 1
    return compute_conf_matrix(predictions, label)


# function that computes the Bayes risk from the confusion matrix given the llrs and the the true labels
def DCF_unorm(llr, label, pi, Cfn, Cfp):
    conf_matrix = compare_with_thre(llr, label, pi, Cfn, Cfp)
    FNR = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1])
    FPR = conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[0][0])
    B_risk = (pi * Cfn * FNR) + ((1 - pi) * Cfp * FPR)
    print(B_risk)
    return B_risk


# function that computes the normmalized Bayes risk from the confusion matrix given the llrs and the true labels
def DCF_norm(llr, label, pi, Cfn, Cfp):
    B_risk = DCF_unorm(llr, label, pi, Cfn, Cfp)
    B_risk_norm = B_risk / min([pi * Cfn, (1 - pi) * Cfp])
    print(B_risk_norm)
    return B_risk_norm


# function that computes the normalized DCF given the confusion matrix
def DCF_norm_conf_matr(conf_matrix, pi, Cfp, Cfn):
    FNR = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1])
    FPR = conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[0][0])
    return (pi * Cfn * FNR + (1 - pi) * Cfp * FPR) / min([pi * Cfn, (1 - pi) * Cfp])


def minimun_DCF_norm(llr, label, pi, Cfp, Cfn):
    # .ravel() transforms the 2D array into a flatterned cont array
    llr = llr.ravel()
    # we need a matrix of all possible thresholds
    thresholds = np.sort(llr)
    DCF = np.zeros(thresholds.shape[0])
    for idx, t in enumerate(thresholds):
        pred = (llr > t) * 1
        conf_mat = compute_conf_matrix(pred, label)
        DCF[idx] = DCF_norm_conf_matr(conf_mat, pi, Cfp, Cfn)
    # argmin returns the indice of the minimum argument
    argmin = DCF.argmin()
    return DCF[argmin], thresholds[argmin]


def roc_points(llr, label):
    llr = llr.ravel()
    thresholds = np.sort(llr)

    # check the number of labels with true label = 0
    # these are the den of TPR and FPR they are the same no matter t
    num_0 = (label == 0).sum()
    num_1 = (label == 1).sum()

    ROC_points_TPR = np.zeros(
        label.shape[0]
    )  # should be filled with value of TPR corresponding to a specific t
    ROC_points_FPR = np.zeros(label.shape[0])
    for idx, t in enumerate(thresholds):
        predicted = (llr > t) * 1
        conf_mat = compute_conf_matrix(predicted, label)
        TPR = conf_mat[1][1] / num_1
        FPR = conf_mat[1][0] / num_0
        ROC_points_TPR[idx] = TPR
        ROC_points_FPR[idx] = FPR
    return ROC_points_TPR, ROC_points_FPR
