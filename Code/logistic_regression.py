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

        dJ = np.concatenate((dJw,np.array(dJb).reshape(1,),))

        return J, dJ

    return logreg_obj


# Train a linear logistic regression model and evaluate it on test data
def linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp):
    # Define objective function
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, pi_T)
    # Optimize objective function
    optV, _, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=False)

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
    logreg_obj = logreg_obj_wrap(phi, LTR, l, pi_T)
    optV, _, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(phi.shape[0] + 1), approx_grad=False)
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


if __name__ == "__main__":
    ### Train and evaluate different logistic regression models using cross validation and single split

    for LR_type in ["linear", "quadratic"]:
        D, L = load("../Train.txt")
        DN = Z_score(D)
        lambda_values = [0, 1e-6, 1e-4, 1e-2, 1, 100]
        pca_values = list(range(5, 11))
        pi_T = 0.5  # to change later
        img_name = "LR_lambda_kfold.png"
        fileName = "../Results/LR_results.txt"
        linear_or_quadratic = linear_logistic_regression
        plot_title = "Linear regression"
        if LR_type == "quadratic":
            fileName = "../Results/Quad_LR_results.txt"
            linear_or_quadratic = quadratic_logistic_regression
            img_name = "Quad_LR_lambda_kfold.png"
            plot_title = "Quadratic regression"

        with open(fileName, "w") as f:
            ### Raw features
            f.write("Values of min DCF for values of lambda = [0, 1e-6, 1e-4, 1e-2, 1, 100]\n")
            f.write("\nRaw features\n")
            DCF_kfold_raw = []
            for l in lambda_values:
                minDCF, _ = k_fold_cross_validation(
                    D, L, linear_or_quadratic, 5, 0.5, 10, 1, l, pi_T, seed=0
                )
                DCF_kfold_raw.append(minDCF)
                f.write(f"Lambda: {l} minDCF: {minDCF}\n")

            ### Z-normalized features - no PCA
            f.write("\nZ-normalized features - no PCA\n")
            DCF_kfold_z = []
            for l in lambda_values:
                minDCF, _ = k_fold_cross_validation(
                    DN, L, linear_or_quadratic, 5, 0.5, 10, 1, l, pi_T, seed=0
                )
                DCF_kfold_z.append(minDCF)
                f.write(f"Lambda: {l} minDCF: {minDCF}\n")

            ### Z-normalized features - PCA = 9 ( chose our pca value)
            f.write("\nZ-normalized features - PCA = 9\n")
            DCF_kfold_z_pca = {k: list() for k in pca_values}
            for l in lambda_values:
                for i in pca_values:
                    preprocessor = partial(PCA_preproccessor, dim=i)
                    minDCF, _ = k_fold_cross_validation(D, L, linear_or_quadratic, 5, 0.5, 10, 1, l, 
                                                        pi_T, preprocessor=preprocessor)
                    DCF_kfold_z_pca[i].append(minDCF)
                    f.write(f"Lambda: {l} PCA dim:{i} minDCF {minDCF}\n")

            ### Plot min DCF for different values of lambda
            plt.figure()
            plt.plot(lambda_values, DCF_kfold_raw, label="Raw")
            plt.plot(lambda_values, DCF_kfold_z, label="Z-normalized")
            for i in pca_values:
                plt.plot(lambda_values, DCF_kfold_z_pca[i], label=f"Z-normalized, PCA {i}")
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend()
            plt.title(plot_title)
            plt.savefig("../Visualization/" + img_name)
