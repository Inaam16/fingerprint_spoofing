import numpy as np
from Metrics import act_DCF, optimal_Bayes_decisions, min_DCF, confusion_matrix, Bayes_risk
import logistic_regression as lr
from Utilities import load
import GMM
import SVM
from MLCore import Z_score, Z_score_eval, PCA, PCA_preproccessor
import matplotlib.pyplot as plt

if __name__ == "__main__":


    DTR, LTR = load("./Train.txt")
    DTE, LTE = load("./Test.txt")  


    # Z-score normalization
    DNTR = Z_score(DTR)
    DNTE = Z_score_eval(DTR, DTE)

    #PCA = 6
    DTR_6, _, DTE_6, _ =  PCA_preproccessor(DTR, LTR, DTE, LTE, 6)
    DNTR_6, _, DNTE_6, _ =  PCA_preproccessor(DNTR, LTR, DNTE, LTE, 6)
    pi = 1/11
    pi_T = 1/11
    Cfn = 1
    Cfp = 1
    K_SVM = 1
    
    ### Evaluating LR
    filename = "./Results/Evaluation/Eval_LR_8.txt"

    lambda_values = [0, 1e-6, 1e-4, 1e-2, 1, 100]
    with open(filename, "w") as f:
        for pca in [8]:
            f.write(f"{pca} \n")
            if pca == 6:
                DTR = DTR_6
                DTE = DTE_6
            else:
                DTR = DTR
                DTE = DTE
            for l in lambda_values:
                
                _, minDCF_lr = lr.quadratic_logistic_regression(DTR_6, LTR, DTE_6, LTE, l, 1/11, 1/11, 1, 1)
                f.write(f"lambda : {l} , minDCF: {minDCF_lr} \n ")
    print(minDCF_lr)




    ### EVAL SVM ###
    filename = "./Results/Evaluation/Eval_SVM_poly.txt"


    C_val = [1e-5, 1e-4, 1e-2 ,1e-1, 1, 10 ]
    with open(filename, "w") as f:
        for svm_k in ["RBF", "poly"]:
            f.write(f"{svm_k} \n")
            for K in [0,1]:
                f.write(f"K: {K} \n")
                for C in C_val:
                    _, minDCF_svm = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, C, svm_k, 1/11, 1, 1, 1/11, d=2, csi = K**0.5, rebalancing=False , c=1)
                    f.write(f"C: {C}, minDCF: {minDCF_svm} \n ")
                    print(minDCF_svm)

    # _, minDCF_gmm = GMM.GMM_classifier_1(DTR, LTR, DTE, LTE, 2, 8, 2, True, False, True, False, 1/11, 1, 1)
    # print(minDCF_gmm)


