import numpy as np
from Metrics import act_DCF, optimal_Bayes_decisions, min_DCF, confusion_matrix, Bayes_risk
import logistic_regression as lr
from Utilities import load
import GMM
import SVM
from MLCore import Z_score, Z_score_eval, PCA, PCA_preproccessor
import matplotlib.pyplot as plt

import calibration

if __name__ == "__main__":


    DTR, LTR = load("../Train.txt")
    DTE, LTE = load("../Test.txt")  


    # Z-score normalization
    DNTR = Z_score(DTR)
    DNTE = Z_score_eval(DTR, DTE)

    #PCA = 6
    DTR_6, _, DTE_6, _ =  PCA_preproccessor(DTR, LTR, DTE, LTE, 6)
    DNTR_6, _, DNTE_6, _ =  PCA_preproccessor(DNTR, LTR, DNTE, LTE, 6)
    
    #PCA = 8
    DTR_8, _, DTE_8, _ =  PCA_preproccessor(DTR, LTR, DTE, LTE, 8)
    DNTR_8, _, DNTE_8, _ =  PCA_preproccessor(DNTR, LTR, DNTE, LTE, 8  )
    
    pi = 1/11
    pi_T = 1/11
    Cfn = 1
    Cfp = 1
    K_SVM = 1
    
    ### Evaluating LR
    # filename = "../Results/Evaluation/Eval_LR_8.txt"

    # lambda_values = [0, 1e-6, 1e-4, 1e-2, 1, 100]
    # with open(filename, "w") as f:
    #     for pca in [8]:
    #         f.write(f"{pca} \n")
    #         if pca == 8:
    #             DTR = DTR_8
    #             DTE = DTE_8
    #         else:
    #             DTR = DTR
    #             DTE = DTE
    #         for l in lambda_values:
    #             _, minDCF_lr = lr.quadratic_logistic_regression(DTR, LTR, DTE, LTE, l, 1/11, 1/11, 1, 1)
    #             f.write(f"lambda : {l} , minDCF: {minDCF_lr} \n ")
    #             print(minDCF_lr)



    ### EVAL SVM Polynomial ###
    # filename = "../Results/Evaluation/Eval_SVM_quad_pca6.txt"
    # C_val = [ 1e-4, 1e-2 ,1e-1, 1, 10 ]
    
    # with open(filename, "w") as f:
    #         svm_k="poly"
    #         f.write(f"{svm_k} \n")
    #         #if svm_k == "poly":
    #         for K in [0,1]:
    #             f.write(f"K: {K} \n")
    #             for C in C_val:
    #                 _, minDCF_svm = SVM.kernel_SVM(DNTR_6, LTR, DNTE_6, LTE, C, svm_k, 1/11, 1, 1, 1/11, d=2, csi = K**0.5, rebalancing=False , c=1)
    #                 f.write(f"C: {C}, minDCF: {minDCF_svm} \n ")
    #                 print(minDCF_svm)
    
                
    ### EVAL SVM  RBF ###
    # filename_rbf= "../Results/Evaluation/Eval_SVM_rbf_all_norm.txt"
    # C_val = [ 1e-4, 1e-2 ,1e-1, 1, 10 ]     
    # gamma_val = [np.exp(-2), np.exp(-3),np.exp(-4)]
    # with open(filename_rbf, "w") as f:
    #         svm_k="RBF"
    #         f.write(f"{svm_k} \n")
    #         for pca in [6, None]:
    #               f.write(f"{pca} \n")
    #               if pca == 6:
    #                   DT = DNTR_6
    #                   DE = DNTE_6
    #               else:
    #                   DT = DNTR
    #                   DE = DNTE
    #               for g in gamma_val:
    #                   print(f"gamma: {g} \n")
    #                   K = 1
    #                   for C in C_val:
    #                       print(f"C: {C} \n")
    #                       _, minDCF_svm = SVM.kernel_SVM(DT, LTR, DE, LTE, C, svm_k, 1/11, 1, 1, 1/11, d=2, csi = K**0.5, rebalancing=False , c=1, gamma=g)
    #                       f.write(f"C: {C}, minDCF: {minDCF_svm} \n ")
    #                       print(minDCF_svm)





    ### EVAL GMM ###
    # filename = "../Results/Evaluation/Eval_GMM"

    # with open(filename, "w") as f:
 
    #     for pca in [6, None]:
    #         f.write(f"PCA: {pca} \n")
    #         if pca == 6:
    #             DT = DTR_6
    #             DE = DTE_6
    #         else:
    #             DT = DTR
    #             DE = DTE
    #         for tied0 in [True, False]:
    #             f.write(f"Tied0: {tied0}")
    #             for tied1 in [True, False]:
    #                 f.write(f"Tied1: {tied1}")
    #                 for n0 in [4,8]:
    #                     _, minDCF_gmm = GMM.GMM_classifier_1(DT, LTR, DE, LTE, 2, n0, 2, True, tied0, True, tied1, 1/11, 1, 1)
    #                     f.write(f"({n0},2) min DCF: {minDCF_gmm} \n")
    #                     print(minDCF_gmm)

        


    
    ### Calibration for best evaluation models ###

    #Best model Quadratic LR:
    # lamda = 0
    # PCA = 6

    # llrLR , _ = lr.quadratic_logistic_regression(DTR_6, LTR, DTE_6, LTE, 0, 1/11, 1/11, 1, 1)
    # np.save("llrLR_eval_cal.npy", llrLR)
    # llrLR = np.load("./llrLR_eval_cal.npy")
    # llrLRcal = calibration.analyse_scores_kfold(llrLR, 1/11, 1, 1, LTE, 5, 1/11,  "LR calibrated")
    # calibration.Bayes_error_plots(llrLRcal, LTE, "LR_calibrated")

    # llrLR , _ = lr.quadratic_logistic_regression(DTR_6, LTR, DTE_6, LTE, 0, 1/11, 1/11, 1, 1)
    # np.save("llrLR_eval.npy", llrLR)
    # llrLR = np.load("./llrLR_eval.npy")
    # llrLRcal = calibration.analyse_scores_kfold(llrLR, 1/11, 1, 1, LTE, 5, 1/11,  "LR_eval")
    # calibration.Bayes_error_plots(llrLR, LTE, "LR_eval")



    #Best model GMM:
    # True True True False
    # no pca
    # 8,2

    # llrGMM, _ = GMM.GMM_classifier_1(DTR, LTR, DTE, LTE, 2, 8, 2, True, True, True, False, 1/11, 1, 1)
    # np.save("llrGMM_eval_cal.npy", llrGMM)
    # llrGMM = np.load("./llrGMM_eval_cal.npy")
    # llrGMMcal = calibration.analyse_scores_kfold(llrGMM, 1/11, 1, 1, LTE, 5, 1/11,  "GMM_eval")
    # calibration.Bayes_error_plots(llrGMM, LTE, "GMM_eval")



    # llrGMM, _ = GMM.GMM_classifier_1(DTR, LTR, DTE, LTE, 2, 8, 2, True, True, True, False, 1/11, 1, 1)
    # np.save("llrGMM_eval_cal.npy", llrGMM)
    # llrGMM = np.load("./llrGMM_eval_cal.npy")
    # llrGMMcal = calibration.analyse_scores_kfold(llrGMM, 1/11, 1, 1, LTE, 5, 1/11,  "GMM calibrated_eval")
    # calibration.Bayes_error_plots(llrGMMcal, LTE, "GMM_calibrated_eval")


    #Best model SVM 
    #RBF kernel SVM, C=10 , logGamma -3,noPCA, Znorm=true, Rebal=false,K=1
    K=1
    

    #llrSVM, _ = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, 10, "RBF",1/11, 1, 1, 1/11, d=2, csi = K**0.5, rebalancing=False , c=1, gamma=np.exp(-3))
    # np.save("llrSVM_eval.npy",llrSVM)
    # llrSVM= np.load("./llrSVM_eval.npy")
    #  llrSVMcal = calibration.analyse_scores_kfold(llrSVM, 1/11, 1, 1, LTE, 5, 1/11,  "SVM_eval")
    # calibration.Bayes_error_plots(llrSVM, LTE, "SVM_eval")
  

    # llrSVM, _ = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, 10, "RBF",1/11, 1, 1, 1/11, d=2, csi = K**0.5, rebalancing=False , c=1, gamma=np.exp(-3))
    # np.save("llrSVM_eval_cal.npy",llrSVM)
    # llrSVM= np.load("./llrSVM_eval_cal.npy")
    # llrSVMcal = calibration.analyse_scores_kfold(llrSVM, 1/11, 1, 1, LTE, 5, 1/11,  "SVM calibrated")
    # calibration.Bayes_error_plots(llrSVMcal, LTE, "SVM_calibrated")



    # images were moved to the ./Results/Evaluation folder

  

    ##Evaluation:
    # _, minDCF_svm = SVM.kernel_SVM(DNTR_6, LTR, DNTE_6, LTE, 0.01, "poly", 1/11, 1, 1, 1/11, d=2, csi = 1**0.5, rebalancing=False , c=1)
    # print(minDCF_svm)
    
    _, minDCF_gmm = GMM.GMM_classifier_1(DTR, LTR, DTE, LTE, 2, 8, 2, True, False, True, False, 0.2, 1, 1)
    print(minDCF_gmm)















