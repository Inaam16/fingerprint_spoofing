import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from Utilities import load
from Metrics import min_DCF
from MLCore import Z_score
import matplotlib.pyplot as plt
from MLCore import PCA_preproccessor

from fractions import Fraction


def objective_function_wrapper(H_hat):
    """Compute the objective function and its gradient"""
    def obj_function_gradient(alpha):
        obj_function = 0.5 * np.dot(np.dot(alpha.T, H_hat), alpha) - sum(alpha)
        gradient = np.dot(H_hat, alpha) - 1
        return obj_function, gradient.reshape([gradient.size, ])

    return obj_function_gradient

# Compute the kernel dot-product
def kernel(x1, x2, kernel_type, d=0, c=0, gamma=0, csi=0):

    if kernel_type.lower() == "poly":
        # Polynomial kernel of degree d
        return (np.dot(x1.T, x2) + c) ** d + csi
    elif kernel_type.lower() == "rbf":
        # RBF kernel
        k = np.zeros([x1.shape[1], x2.shape[1]])
        for index1 in range(x1.shape[1]):
            for index2 in range(x2.shape[1]):
                k[index1, index2] = np.exp(-gamma * ((x1[:, index1] - x2[:, index2]) * (x1[:, index1] - x2[:, index2])).sum()) + csi
        return k
    else:
        raise ValueError("Kernel type must be either 'poly' or 'rbf'")

# Train a linear SVM model and evaluate it on test data
def linear_SVM(DTR, LTR, DTE, LTE, C, K, pi, Cfp, Cfn, pi_T, rebalancing=True):
    
    D_hat = np.concatenate((DTR, K * np.array(np.ones([1, DTR.shape[1]]))))
    
    # Compute H_hat
    Z = np.ones(LTR.shape)
    Z[LTR == 0] = -1
    ZiZj = np.dot(Z.reshape([Z.shape[0], 1]), Z.reshape([Z.shape[0], 1]).T)
    H_hat = ZiZj * np.dot(D_hat.T, D_hat)

    # Define the objective function and the bounds
    obj_function_gradient = objective_function_wrapper(H_hat)
    B = np.zeros([DTR.shape[1], 2])
    if rebalancing:
        # To rebalance classes, use two different values of C based on the empirical prior
        pi_T_emp = sum(LTR == 1) / DTR.shape[1]
        Ct = C * pi_T / pi_T_emp
        Cf = C * (1 - pi_T) / (1 - pi_T_emp)
        B[LTR == 1, 1] = Ct
        B[LTR == 0, 1] = Cf
    else:
        B[:, 1] = C
    # Optimize the objective function
    optAlpha, _, _ = fmin_l_bfgs_b(obj_function_gradient, np.zeros(DTR.shape[1]),
                                   approx_grad=False, bounds=B, factr=10000.0)

    # Recover primal solution
    w_hat = np.sum(optAlpha * Z * D_hat, axis=1)

    # Compute extended data matrix for test set
    T_hat = np.concatenate((DTE, K * np.array(np.ones([1, DTE.shape[1]]))))

    # Compute scores and minDCF
    S = np.dot(w_hat.T, T_hat)
    minDCF, _ = min_DCF(S, pi, Cfn, Cfp, LTE)

    return S, minDCF

# Train a kernel SVM model and evaluate it on test data
def kernel_SVM(DTR, LTR, DTE, LTE, C, type, pi, Cfn, Cfp, pi_T, d=0, c=0, gamma=0, csi=0, rebalancing=True):

    # Compute H_hat
    Z = np.ones(LTR.shape)
    Z[LTR == 0] = -1
    ZiZj = np.dot(Z.reshape([Z.shape[0], 1]), Z.reshape([Z.shape[0], 1]).T)
    H_hat = ZiZj * kernel(DTR, DTR, type, d, c, gamma, csi)

    # Define the objective function and the bounds
    obj_function_gradient = objective_function_wrapper(H_hat)
    B = np.zeros([DTR.shape[1], 2])
    if rebalancing:
        # To rebalance classes, use two different values of C based on the empirical prior
        pi_T_emp = sum(LTR == 1) / DTR.shape[1]
        Ct = C * pi_T / pi_T_emp
        Cf = C * (1 - pi_T) / (1 - pi_T_emp)
        B[LTR == 1, 1] = Ct
        B[LTR == 0, 1] = Cf
    else:
        B[:, 1] = C
    # Optimize the objective function
    optAlpha, _, _ = fmin_l_bfgs_b(obj_function_gradient, np.zeros(DTR.shape[1]),
                                   approx_grad = False, bounds = B, factr = 10000.0)

    # Compute scores and minDCF
    S = np.sum((optAlpha * Z).reshape([DTR.shape[1], 1]) * kernel(DTR, DTE, type, d, c, gamma, csi), axis = 0)
    minDCF, _ = min_DCF(S, pi, Cfn, Cfp, LTE)

    return S, minDCF

# Perform k-fold cross validation on test data for the specified model
def k_fold_cross_validation(D, L, classifier, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = True, 
                            gamma = 0, seed = 0, kernel_type = "", pca_dim=None):

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

    llr = np.zeros([D.shape[1], ])

    for count in range(k):
        # Define training and test partitions
        if start_index + elements > D.shape[1]:
            end_index = D.shape[1]
        else:
            end_index = start_index + elements 

        idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
    
        DTE = D[:, idxTest]
        LTE = L[idxTest]

        if pca_dim:
            DTR, LTR, DTE, LTE = PCA_preproccessor(DTR, LTR, DTE, LTE, pca_dim)

        # Train the classifier and compute llr on the current partition
        if kernel_type == "": # linear SVM
            llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing)
        else: # kernel SVM
            llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, C, kernel_type, pi, Cfn, Cfp, pi_T, gamma=gamma,
                                     rebalancing=rebalancing, d =3, csi=K_SVM**0.5, c = 1)

        start_index += elements

    # Evaluate results after all k-fold iterations (when all llr are available)
    minDCF_value, _ = min_DCF(llr, pi, Cfn, Cfp, L)
    return minDCF_value, llr










if __name__ == "__main__":
    ### Train and evaluate different SVM models using cross validatio
    ### Save results in a file and print figures for hyperparameters estimation


    # results_SVM_linear(5,Fraction(1,11), 1, 1, Fraction(1,11), 6, False, 1, title="pi1")

    D, L = load("../Train.txt")
    DN = Z_score(D)
    C_val = [1e-5, 1e-4, 1e-2 ,1e-1, 1, 10 ]
    pi, pi_T = 1/11, 1/11
    Cfn, Cfp = 1, 1
    folds = 5
    K_SVM = 1
    pca_dim = 6

    ### LINEAR SVM
    # fileName = "../Results/linear_SVM_results.txt"
    # linear_or_quadratic = linear_SVM

    # with open(fileName, "w") as f:
    #     f.write("**** min DCF for different linear SVM models ****\n\n")
    #     f.write("Values of min DCF for values of C \n")

    #     for rebalancing in [True,False]:

    #         f.write("\n Rebalancing: " + str(rebalancing) + "\n")

    #         f.write("\nRaw features\n")
    #         DCF_kfold_raw = []
    #         for C in C_val:
    #             minDCF, _ = k_fold_cross_validation(D, L, linear_or_quadratic, folds, pi, Cfp, Cfn, C, pi_T, K_SVM,pca_dim=pca_dim, rebalancing=rebalancing)
    #             DCF_kfold_raw.append(minDCF)
    #             f.write("5-fold: " + str(minDCF))

    #         print("Finished raw features")

    #         f.write("\nZ-normalized features - no PCA\n")
    #         DCF_kfold_z = []
    #         for C in C_val:
    #             minDCF, _ = k_fold_cross_validation(DN, L, linear_or_quadratic, folds, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing=rebalancing)
    #             DCF_kfold_z.append(minDCF)
    #             f.write("5-fold: " + str(minDCF))

    #         print("Finished Z-normalized features")

    #         img_name = f"SVM_C_kfold_{'rebal' if rebalancing else 'norebal'}.png"

    #         plt.figure()
    #         plt.plot(C_val, DCF_kfold_raw, label='Raw')
    #         plt.plot(C_val, DCF_kfold_z, label='Z-normalized')
    #         plt.xscale("log")
    #         plt.xlabel(r"$C$")
    #         plt.ylabel("min DCF")
    #         plt.legend()
    #         plt.savefig("../Results/SVM/" + img_name)
    #         plt.close()

    # ### QUADRATIC KERNEL SVM
    
    # fileName = "../Results/SVM/quad_SVM_results.txt"
    # with open(fileName, "w") as f:
        
    #     f.write("**** min DCF for different quadratic kernel SVM models ****\n\n")
    #     f.write("Values of min DCF for values of C = [1e-1, 1, 10]\n")

    #     f.write("\nZ-normalized features -  PCA 6  - no rebalancing\n")
    #     DCF_kfold_z_nobal = []
    #     for C in C_val:
    #         minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, folds, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = False,pca_dim=6, kernel_type= "poly")
    #         DCF_kfold_z_nobal.append(minDCF)
    #         f.write("5-fold: " + str(minDCF))
        
    #     print("Finished Z-normalized features - no rebalancing")

    #     f.write("\nZ-normalized features -  PCA 6 - rebalancing\n")
    #     DCF_kfold_z_bal = []
    #     for C in C_val:
    #         minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, folds, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = True, pca_dim=6, kernel_type = "poly")
    #         DCF_kfold_z_bal.append(minDCF)
    #         f.write("5-fold: " + str(minDCF))
    #     print("Finished Z-normalized features - rebalancing")
        
    #     f.write("\nZ-normalized features - no PCA - no rebalancing\n")
    #     DCF_kfold_z_nobal_noPCA = []
    #     for C in C_val:
    #         minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, folds, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = False,pca_dim=None, kernel_type= "poly")
    #         DCF_kfold_z_nobal_noPCA.append(minDCF)
    #         f.write("5-fold: " + str(minDCF))
        
    #     print("Finished Z-normalized features - no rebalancing")

    #     f.write("\nZ-normalized features - no PCA - rebalancing\n")
    #     DCF_kfold_z_bal_noPCA = []
    #     for C in C_val:
    #         minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, folds, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = True, pca_dim=None, kernel_type = "poly")
    #         DCF_kfold_z_bal_noPCA.append(minDCF)
    #         f.write("5-fold: " + str(minDCF))
    #     print("Finished Z-normalized features - rebalancing")

    #     img_name = "quad_3_SVM_C_kfold.png"

    #     plt.figure()
    #     plt.plot(C_val, DCF_kfold_z_nobal, marker='o', label='PCA 6 No balancing')
    #     plt.plot(C_val, DCF_kfold_z_bal, marker='o', label='PCA 6 Rebalancing')
    #     plt.plot(C_val, DCF_kfold_z_nobal_noPCA, marker='o', label='no PCA No balancing')
    #     plt.plot(C_val, DCF_kfold_z_bal_noPCA, marker='o', label='no PCA  Rebalancing')
    #     plt.xscale("log")
    #     plt.xlabel("C")
    #     plt.ylabel("min DCF")
    #     plt.legend()
    #     plt.savefig("../Results/SVM/" + img_name)
        
        
    
    ### RBF KERNEL SVM
    
    fileName = "../Results/SVM/RBF_SVM_results_PCA6.txt"
    gamma_val = [np.exp(-2), np.exp(-3),np.exp(-4)]

    with open(fileName, "w") as f:

        # DCF_kfold_z_nobal_noPCA = np.zeros([len(gamma_val), len(C_val)])
        DCF_kfold_z_nobal_PCA6 = np.zeros([len(gamma_val), len(C_val)])

        for i, gamma in enumerate(gamma_val):
            f.write("**** min DCF for different quadratic kernel SVM models ****\n\n")
            f.write("Values of min DCF for values of C \n")

            # f.write("\nZ-normalized features - no PCA - no rebalancing\n")
            # for j, C in enumerate(C_val):
            #     minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, folds, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = False, type = "RBF", gamma = gamma)
            #     DCF_kfold_z_nobal_noPCA[i, j] = (minDCF)
            #     f.write("5-fold: " + str(minDCF))
            
            # print("Finished Z-normalized features - no rebalancing")

            f.write("\nZ-normalized features -  PCA 6 - no rebalancing\n")
            for j, C in enumerate(C_val):
                minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, folds, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = False,pca_dim=6, kernel_type = "RBF", gamma = gamma)
                DCF_kfold_z_nobal_PCA6[i, j] = (minDCF)
                f.write("5-fold: " + str(minDCF))
            
            print("Finished Z-normalized features PC6 - no rebalancing")
          

        img_name = "RBF_SVM_C_kfold_nobal_PCA6.png"

        plt.figure()
        # plt.plot(C_val, DCF_kfold_z_nobal_noPCA[0,:])
        plt.plot(C_val, DCF_kfold_z_nobal_PCA6[0,:])
     
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("min DCF")
        plt.legend([r"$log \gamma = 2$",
            r"$log \gamma = -3$", r"$log \gamma = -4$"])
        plt.savefig("../Results/SVM/" + img_name)
