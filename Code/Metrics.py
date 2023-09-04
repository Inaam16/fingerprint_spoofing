import numpy


# compute the conf matrix given the predictions and the true labels
def compute_conf_matrix(predictLabel, Label):
    conf_matrix = numpy.zeros((2,2))
    for i in range(2):
        for j in range(2):
            conf_matrix[i,j] = ( 1 * numpy.bitwise_and(predictLabel == i, Label== j)).sum()
    return conf_matrix


# comapre the llrs with threshold and return the confusion matrix
def compare_with_thre(llr, label, pi, Cfn, Cfp):
    t = -numpy.log(pi*Cfn/(1-pi)*Cfp)
    predictions = (llr > t)*1
    return compute_conf_matrix(predictions, label)

# function that computes the Bayes risk from the confusion matrix given the llrs and the the true labels
def DCF_unorm(llr, label, pi, Cfn, Cfp):
    conf_matrix = compare_with_thre(llr, label, pi, Cfn, Cfp)
    FNR = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1])
    FPR = conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[0][0])
    B_risk = ((pi * Cfn * FNR) + ((1-pi) * Cfp * FPR))
    print(B_risk)
    return B_risk 

# function that computes the normmalized Bayes risk from the confusion matrix given the llrs and the true labels
def DCF_norm(llr, label, pi, Cfn, Cfp):
    B_risk = DCF_unorm(llr, label , pi, Cfn, Cfp)
    B_risk_norm = B_risk / min([pi*Cfn, (1-pi)*Cfp])
    print(B_risk_norm)
    return B_risk_norm

# function that computes the normalized DCF given the confusion matrix
def DCF_norm_conf_matr(conf_matrix, pi, Cfp, Cfn):
    FNR = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1])
    FPR = conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[0][0])
    return (pi*Cfn*FNR + (1-pi)*Cfp*FPR) / min([pi*Cfn, (1-pi)*Cfp])


def minimun_DCF_norm(llr, label, pi, Cfp, Cfn):
    #.ravel() transforms the 2D array into a flatterned cont array
    llr = llr.ravel()
    #we need a matrix of all possible thresholds
    thresholds = numpy.sort(llr)
    DCF = numpy.zeros(thresholds.shape[0])
    for(idx,t) in enumerate(thresholds):
        pred = (llr > t) * 1
        conf_mat = compute_conf_matrix(pred, label)
        DCF[idx] = DCF_norm_conf_matr(conf_mat, pi, Cfp, Cfn)
    #argmin returns the indice of the minimum argument
    argmin = DCF.argmin()    
    return DCF[argmin], thresholds[argmin]



def roc_points(llr, label):
    llr = llr.ravel()
    thresholds = numpy.sort(llr)

    #check the number of labels with true label = 0
    #these are the den of TPR and FPR they are the same no matter t
    num_0 = (label == 0).sum()
    num_1 = (label == 1).sum()

    ROC_points_TPR = numpy.zeros(label.shape[0]) #should be filled with value of TPR corresponding to a specific t
    ROC_points_FPR = numpy.zeros(label.shape[0])
    for(idx, t) in enumerate(thresholds):
        predicted = (llr > t) * 1
        conf_mat = compute_conf_matrix(predicted, label)
        TPR = conf_mat[1][1] / num_1
        FPR = conf_mat[1][0] / num_0
        ROC_points_TPR[idx] = TPR
        ROC_points_FPR[idx] = FPR
    return ROC_points_TPR, ROC_points_FPR