"""
LOGISTIC REGRESSION
"""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from Metrics import *
import matplotlib.pyplot as plt
from pre_processing import PCA
from MLCore import *
from GaussianModels import PCA_preproccessor
from functools import partial
from tqdm import tqdm
from fractions import Fraction


def normalize_zscore(D, mu=[], sigma=[]):
   # print("D shape: "+str(D.shape))
    if mu == [] or sigma == []:
        mu = np.mean(D, axis=1)
        sigma = np.std(D, axis=1)
    ZD = D
    ZD = ZD - mCol(mu)
    ZD = ZD / mCol(sigma)
    #print("ZD shape: "+str(ZD.shape))
    return ZD



# Compute objective function and its derivatives (required by the optimization algorithm)
def logreg_obj_wrap(DTR, LTR, l, pi_T):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]

        Nt = sum(LTR == 1)
        Nf = sum(LTR == 0)

        J = (
            l / 2 * np.linalg.norm(w) ** 2
            + pi_T / Nt * sum(np.log1p(np.exp(-(np.dot(w.T, DTR[:, LTR == 1]) + b))))
            + (1 - pi_T) / Nf * sum(np.log1p(np.exp((np.dot(w.T, DTR[:, LTR == 0]) + b))))
        )

        dJw = (
            l * w
            - pi_T
            / Nt
            * np.sum(DTR[:, LTR == 1] / (1 + np.exp(np.dot(w.T, DTR[:, LTR == 1]) + b)), axis=1)
            + (1 - pi_T)
            / Nf
            * np.sum(DTR[:, LTR == 0] / (1 + np.exp(-np.dot(w.T, DTR[:, LTR == 0]) - b)), axis=1)
        )

        dJb = -pi_T / Nt * np.sum(1 / (1 + np.exp(np.dot(w.T, DTR[:, LTR == 1]) + b))) + (
            1 - pi_T
        ) / Nf * np.sum(1 / (1 + np.exp(-np.dot(w.T, DTR[:, LTR == 0]) - b)))

        dJ = np.concatenate((dJw, np.array(dJb).reshape(1,),))

        return J, dJ

    return logreg_obj


def logreg_obj_wrap_stable(DTR, LTR, l, pi_T):
    def logreg_obj(v):
        w, b = v[:-1], v[-1]

        # Indices for each case
        indices_z1 = LTR == 1  # z = 1
        indices_z_minus_1 = LTR == 0  # z = -1
        Nt = sum(indices_z1)
        Nf = sum(indices_z_minus_1)

        # Compute for z = 1
        sum_z1 = np.sum(np.logaddexp(0, -np.dot(w.T, DTR[:, indices_z1]) - b))
        # Compute for z = -1
        sum_z_minus_1 = np.sum(np.logaddexp(0, np.dot(w.T, DTR[:, indices_z_minus_1]) + b))

        J = (l/2) * np.linalg.norm(w)**2 + (pi_T / Nt) * sum_z1 + ((1-pi_T)/Nf) * sum_z_minus_1

        return J
    return logreg_obj


# Train a linear logistic regression model and evaluate it on test data
def linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp):
    # Define objective function
    logreg_obj = logreg_obj_wrap_stable(DTR, LTR, l, pi_T)
    # Optimize objective function
    optV, _, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True)

    w, b = optV[0:-1], optV[-1]

    # Compute scores
    s = np.dot(w.T, DTE) + b

    minDCF, _ = min_DCF(s, pi, Cfn, Cfp, LTE)

    return s, minDCF


# Map features to the quadratic feature space
def map_to_feature_space(D):
    phi = np.zeros([D.shape[0] ** 2 + D.shape[0], D.shape[1]])
    for index in range(D.shape[1]):
        x = D[:, index].reshape(D.shape[0], 1)
        # phi = [vec(x*x^T), x]^T
        phi[:, index] = np.concatenate((np.dot(x, x.T).reshape(x.shape[0] ** 2, 1), x)).reshape(
            phi.shape[0],
        )
    return phi


# Train a quadratic logistic regression model and evaluate it on test data
def quadratic_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp):
    # Map training features to expanded feature space
    phi = map_to_feature_space(DTR)

    # Train a linear regression model on expanded feature space
    logreg_obj = logreg_obj_wrap_stable(phi, LTR, l, pi_T)
    optV, _, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(phi.shape[0] + 1), approx_grad=True)
    w, b = optV[0:-1], optV[-1]

    # Map test features to expanded feature space
    phi_test = map_to_feature_space(DTE)

    # Compute scores
    s = np.dot(w.T, phi_test) + b
    minDCF, _ = min_DCF(s, pi, Cfn, Cfp, LTE)

    return s, minDCF


# Perform k-fold cross validation on test data for the specified model
def k_fold_cross_validation(
    D, L, classifier, k, pi, Cfp, Cfn, l, pi_T, seed=0, preprocessor=None, just_llr=False
):
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

    llr = np.zeros([D.shape[1],])

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

        if preprocessor:
            DTR, LTR, DTE, LTE = preprocessor(DTR, LTR, DTE, LTE)

        # Train the classifier and compute llr on the current partition
        llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
        start_index += elements

    if just_llr:
        minDCF = 0
    else:
        # Evaluate results after all k-fold iterations (when all llr are available)
        minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, llr

def results_LR_QLR(pi, Cfn, Cfp, znorm, title):
    if znorm == True:
        label_z = "Z_norm"
    else:
        label_z = ""
    for LR_type in tqdm(["linear", "quadratic"], desc='Linear or Quadratic'):
        D, L = load("./Train.txt")
        if znorm == True:
            DN = normalize_zscore(D)
        else:
            DN = D
        lambda_values = [0, 1e-6, 1e-4, 1e-2, 1, 100]
        pca_values = list(range(6, 11))
        pi_T = float(Fraction(1,11)) # to change later
        img_name = f"LR_lambda_kfold_{title}_{label_z}.png"
        fileName = f"./Results/LR/LR_results_{title}_{label_z}.txt"
        linear_or_quadratic = linear_logistic_regression
        plot_title = "Linear regression"
        if LR_type == "quadratic":
            fileName = f"./Results/LR/Quad_LR_results_{title}_{label_z}.txt"
            linear_or_quadratic = quadratic_logistic_regression
            img_name = f"Quad_LR_lambda_kfold_{title}_{label_z}.png"
            plot_title = "Quadratic regression"

        with open(fileName, "w") as f:
            ### Raw features
            f.write("Values of min DCF for values of lambda = [0, 1e-6, 1e-4, 1e-2, 1, 100]\n")
            f.write("\nRaw features\n")
            DCF_kfold_raw = []
            for l in tqdm(lambda_values, desc='Un-normalized'):
                minDCF, _ = k_fold_cross_validation(
                    DN, L, linear_or_quadratic, 5, pi, Cfp, Cfn, l, pi_T, seed=0
                )
                DCF_kfold_raw.append(minDCF)
                f.write(f"Lambda: {l} minDCF: {minDCF}\n")

            ### Z-normalized features - no PCA
            # f.write("\nZ-normalized features - no PCA\n")
            # DCF_kfold_z = []
            # for l in tqdm(lambda_values, desc='Normalized'):
            #     minDCF, _ = k_fold_cross_validation(
            #         DN, L, linear_or_quadratic, 5, pi, Cfp, Cfn, l, pi_T, seed=0
            #     )
            #     DCF_kfold_z.append(minDCF)
            #     f.write(f"Lambda: {l} minDCF: {minDCF}\n")

            ### Z-normalized features - PCA = 9 ( chose our pca value)
            f.write("\nZ-normalized features - PCA \n")
            DCF_kfold_z_pca = {k: list() for k in pca_values}
            for l in tqdm(lambda_values, desc='With PCA'):
                for i in tqdm(pca_values, desc='PCA values'):
                    preprocessor = partial(PCA_preproccessor, dim=i)
                    minDCF, _ = k_fold_cross_validation(DN, L, linear_or_quadratic, 5, pi, Cfp, Cfn, l, 
                                                        pi_T, preprocessor=preprocessor)
                    DCF_kfold_z_pca[i].append(minDCF)
                    f.write(f"Lambda: {l} PCA dim:{i} minDCF {minDCF}\n")
            if label_z == "Z_norm":
                label_z = "Z-norm"
            ### Plot min DCF for different values of lambda
            plt.figure()
            plt.plot(lambda_values, DCF_kfold_raw, label=f"{label_z} No PCA")
            # plt.plot(lambda_values, DCF_kfold_z, label=f"{label_z}")
            for i in pca_values:
                plt.plot(lambda_values, DCF_kfold_z_pca[i], label=f"{label_z} PCA {i}")
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend()
            plt.title(plot_title)
            plt.savefig("./Results/LR/" + img_name)
    return








def results_LR_QLR_piT(pi_T, znorm):
    pi = float(Fraction(1,11))
    Cfn = 1
    Cfp = 1

    title = str(pi_T)

    if znorm == True:
        label_z = "Z_norm"
    else:
        label_z = ""
    for LR_type in tqdm(["linear", "quadratic"], desc='Linear or Quadratic'):
        D, L = load("./Train.txt")
        if znorm == True:
            DN = normalize_zscore(D)
        else:
            DN = D
        lambda_values = [0, 1e-6, 1e-4, 1e-2, 1, 100]
        pca_values = list(range(6, 10))
        pi_T = pi_T # to change later
        img_name = f"LR_lambda_kfold_{title}_{label_z}.png"
        fileName = f"./Results/LR/piT/LR_results_{title}_{label_z}.txt"
        linear_or_quadratic = linear_logistic_regression
        plot_title = "Linear regression"
        if LR_type == "quadratic":
            fileName = f"./Results/LR/piT/Quad_LR_results_{title}_{label_z}.txt"
            linear_or_quadratic = quadratic_logistic_regression
            img_name = f"Quad_LR_lambda_kfold_{title}_{label_z}.png"
            plot_title = "Quadratic regression"

        with open(fileName, "w") as f:
            ### Raw features
            f.write("Values of min DCF for values of lambda = [0, 1e-6, 1e-4, 1e-2, 1, 100]\n")
            f.write("\nRaw features\n")
            DCF_kfold_raw = []
            for l in tqdm(lambda_values, desc='Un-normalized'):
                minDCF, _ = k_fold_cross_validation(
                    DN, L, linear_or_quadratic, 5, pi, Cfp, Cfn, l, pi_T, seed=0
                )
                DCF_kfold_raw.append(minDCF)
                f.write(f"Lambda: {l} minDCF: {minDCF}\n")

            ### Z-normalized features - no PCA
            # f.write("\nZ-normalized features - no PCA\n")
            # DCF_kfold_z = []
            # for l in tqdm(lambda_values, desc='Normalized'):
            #     minDCF, _ = k_fold_cross_validation(
            #         DN, L, linear_or_quadratic, 5, pi, Cfp, Cfn, l, pi_T, seed=0
            #     )
            #     DCF_kfold_z.append(minDCF)
            #     f.write(f"Lambda: {l} minDCF: {minDCF}\n")

            ### Z-normalized features - PCA = 9 ( chose our pca value)
            f.write("\nZ-normalized features - PCA \n")
            DCF_kfold_z_pca = {k: list() for k in pca_values}
            for l in tqdm(lambda_values, desc='With PCA'):
                for i in tqdm(pca_values, desc='PCA values'):
                    preprocessor = partial(PCA_preproccessor, dim=i)
                    minDCF, _ = k_fold_cross_validation(DN, L, linear_or_quadratic, 5, pi, Cfp, Cfn, l, 
                                                        pi_T, preprocessor=preprocessor)
                    DCF_kfold_z_pca[i].append(minDCF)
                    f.write(f"Lambda: {l} PCA dim:{i} minDCF {minDCF}\n")
            if label_z == "Z_norm":
                label_z = "Z-norm"
            ### Plot min DCF for different values of lambda
            plt.figure()
            plt.plot(lambda_values, DCF_kfold_raw, label=f"{label_z} No PCA")
            # plt.plot(lambda_values, DCF_kfold_z, label=f"{label_z}")
            for i in pca_values:
                plt.plot(lambda_values, DCF_kfold_z_pca[i], label=f"{label_z} PCA {i}")
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend()
            plt.title(plot_title)
            plt.savefig("./Results/LR/piT/" + img_name)
    return












if __name__ == "__main__":
    ### Train and evaluate different logistic regression models using cross validation and single split
    
    
    
    
    
    # results_LR_QLR_piT(float(Fraction(1,11)), True)
    # results_LR_QLR_piT(float(Fraction(1,11)), False)
    
    # results_LR_QLR_piT(0.1, True)
    # results_LR_QLR_piT(0.1, False)
    
    # results_LR_QLR_piT(0.5, True)
    # results_LR_QLR_piT(0.5, False)

    # results_LR_QLR_piT(0.2, True)
    # results_LR_QLR_piT(0.2, False)

    results_LR_QLR_piT(0.33, True)
    results_LR_QLR_piT(0.33, False)


    # pi1 = pieffapp = 1/11
    # results_LR_QLR(float(Fraction(1,11)), 1, 1, True, "pi1")
    #results_LR_QLR(Fraction(1,11), 1, 1, False, "pi1")
    
    # pi2 = 0.9
    # results_LR_QLR(0.9, 1, 1, True, "pi2")
   
    #pi3 = 0.1
    # results_LR_QLR(0.1, 1, 1, True, "pi3")
    
   # results_LR_QLR(0.1, 1, 1, False, "pi3")
    
    #pi4 = 0.5
    # results_LR_QLR(0.5, 1, 1, True, "pi4")
   
    # pi5 = 0.4
    # results_LR_QLR(0.4, 1, 1, False, "pi5")
    # results_LR_QLR(0.4, 1, 1, True, "pi5")

    # pi6 = 0.05
    # results_LR_QLR(0.05, 1, 1, False, "pi6")
    # results_LR_QLR(0.05, 1, 1, True, "pi6")

    # pi7 = 0.95
    # results_LR_QLR(0.095, 1, 1, True, "pi7")

    # pi8 = 0.2
    # results_LR_QLR(0.2, 1, 1, True, "pi8")





    

