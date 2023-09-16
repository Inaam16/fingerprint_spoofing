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

    DTR_8, _, DTE_8, _ =  PCA_preproccessor(DTR, LTR, DTE, LTE, 6)
    DNTR_8, _, DNTE_8, _ =  PCA_preproccessor(DNTR, LTR, DNTE, LTE, 6)


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
            if pca == 8:
                DTR = DTR_8
                DTE = DTE_8
            else:
                DTR = DTR
                DTE = DTE
            for l in lambda_values:
                
                _, minDCF_lr = lr.quadratic_logistic_regression(DTR_6, LTR, DTE_6, LTE, l, 1/11, 1/11, 1, 1)
                f.write(f"lambda : {l} , minDCF: {minDCF_lr} \n ")
    print(minDCF_lr)




    ### EVAL SVM ###
    filename = "./Results/Evaluation/Eval_SVM_quad.txt"


    C_val = [1e-5, 1e-4, 1e-2 ,1e-1, 1, 10 ]
    gamma_val = [np.exp(-2), np.exp(-3),np.exp(-4)]
    with open(filename, "w") as f:
        for svm_k in ["RBF", "poly"]:
            f.write(f"{svm_k} \n")
            if svm_k == "poly":
                for K in [0,1]:
                    f.write(f"K: {K} \n")
                    for C in C_val:
                        _, minDCF_svm = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, C, svm_k, 1/11, 1, 1, 1/11, d=2, csi = K**0.5, rebalancing=False , c=1)
                        f.write(f"C: {C}, minDCF: {minDCF_svm} \n ")
                        print(minDCF_svm)
            else:
                for pca in [6, None]:
                    f.write(f"{pca} \n")
                    if pca == 6:
                        DT = DNTR_6
                        DE = DNTE_6
                    else:
                        DT = DNTR
                        DE = DNTE
                    for g in gamma_val:
                        print(f"g: {g}")
                        K = 1
                        for C in C_val:
                            print(f"C: {C}")
                            _, minDCF_svm = SVM.kernel_SVM(DT, LTR, DE, LTE, C, svm_k, 1/11, 1, 1, 1/11, d=2, csi = K**0.5, rebalancing=False , c=1, gamma=g)
                            f.write(f"C: {C}, minDCF: {minDCF_svm} \n ")
                            print(minDCF_svm)


    filename = "./Results/Evaluation/Eval_GMM"

    with open(filename, "w") as f:
        f.write(f"PCA: {pca} \n")
        for pca in [6, None]:
            if pca == 6:
                DT = DTR_6
                DE = DTE_6
            else:
                DT = DTR
                DE = DTE
            for tied0 in [True, False]:
                f.write(f"Tied0: {tied0}")
                for tied1 in [True, False]:
                    f.write(f"Tied1: {tied1}")
                    for n0 in [4,8]:
                        _, minDCF_gmm = GMM.GMM_classifier_1(DTR, LTR, DTE, LTE, 2, n0, 2, True, tied0, True, tied1, 1/11, 1, 1)
                        f.write(f"({n0},2) min DCF: {minDCF_gmm} \n")
                        print(minDCF_gmm)

        


    

