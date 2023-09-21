import numpy as np
from scipy.special import logsumexp
from Utilities import load
from Metrics import min_DCF
from MLCore import Z_score
from pre_processing import PCA_preproccessor
import matplotlib.pyplot as plt
from itertools import product
from GaussianModels import compute_llr


def GAU_logpdf_ND(x, mu, C):
    M = x.shape[0]

    _, logSigma = np.linalg.slogdet(C)

    if x.shape[1] == 1:
        logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x-mu))
    else:
        logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.diagonal(np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x-mu)))

    return logN


def logpdf_GMM(X, gmm):
    S = np.zeros([len(gmm), X.shape[1]])

    for g in range(len(gmm)):
        S[g, :] = GAU_logpdf_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])

    # marginal log densities
    logdens = logsumexp(S, axis=0)
    # posterior distributions
    logGamma = S - logdens
    gamma = np.exp(logGamma)
    return logdens, gamma


# Tune GMM parameters using EM algorithm
def GMM_EM_estimation(X, gmm, t, psi, diag = False, tied = False):

    curr_gmm = gmm
    ll = t + 1
    prev_ll = 0

    # Stop condition on log-likelihood variation
    while abs(ll - prev_ll) >= t:
        # E-step: compute posterior probabilities
        logdens, gamma = logpdf_GMM(X, curr_gmm)
        if prev_ll == 0:
            prev_ll = sum(logdens) / X.shape[1]
        else:
            prev_ll = ll
        # M-step: update model parameters
        Z = np.sum(gamma, axis=1)

        for g in range(len(gmm)):
            # Compute statistics
            F = np.sum(gamma[g] * X, axis=1)
            S = np.dot(gamma[g] * X, X.T)
            mu = (F / Z[g]).reshape([X.shape[0], 1])
            sigma = S / Z[g] - np.dot(mu, mu.T)
            w = Z[g] / sum(Z)

            if diag:
                # Keep only the diagonal of the matrix
                sigma = sigma * np.eye(sigma.shape[0])

            if not tied: # If tied hypothesis, add constraints only at the end
                U, s, _ = np.linalg.svd(sigma)
                # Add constraints on the covariance matrixes to avoid degenerate solutions
                s[s < psi] = psi
                covNew = np.dot(U, s.reshape([s.shape[0], 1]) * U.T)
                curr_gmm[g] = (w, mu, covNew)
            else:  # if tied, constraints are added later
                curr_gmm[g] = (w, mu, sigma)

        if tied:
            # Compute tied covariance matrix
            tot_sigma = np.zeros(curr_gmm[0][2].shape)
            for g in range(len(gmm)):
                tot_sigma += Z[g] * curr_gmm[g][2] 
            tot_sigma /= X.shape[1]
            U, s, _ = np.linalg.svd(tot_sigma)
            # Add constraints on the covariance matrixes to avoid degenerate solutions
            s[s<psi] = psi
            tot_sigma = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
            for g in range(len(gmm)):
                curr_gmm[g][2][:,:] = tot_sigma 

        # Compute log-likelihood of training data
        logdens, _ = logpdf_GMM(X, curr_gmm)
        ll = sum(logdens) / X.shape[1]

    return curr_gmm, ll

# LBG algorithm: from a GMM with G component, train a GMM with 2G components
def LBG(X, gmm, t, alpha, psi, diag, tied):
    new_gmm = []
    for c in gmm:

        # Compute direction along which to move the means
        U, s, _ = np.linalg.svd(c[2])
        d = U[:, 0:1] * s[0]**0.5 * alpha

        # Create two components from the original one
        new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) + d, c[2]))
        new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) - d, c[2]))

    # Tune components using EM algorithm
    gmm, ll = GMM_EM_estimation(X, new_gmm, t, psi, diag, tied)

    return gmm, ll



# Train a GMM classifier (one GMM for each class) and evaluate it on training dat
def GMM_classifier_1(DTR, LTR, DTE, LTE, n_classes, components0, components1, diag0, tied0, diag1, tied1, pi, Cfn, Cfp, t = 1e-6, psi = 0.01, alpha = 0.1, f=1, type=""):
    
    
    S = np.zeros([n_classes, DTE.shape[1]])
    all_gmm = []

    # Repeat until the desired number of components is reached, but analyze also
    # intermediate models with less components
    for count0 in range(int(np.log2(components0))):
       
        # Train one GMM for each class
           
            if count0 == 0:
                # Start from max likelihood solution for one component
                covNew = np.cov(DTR[:, LTR == 0])
                # Impose the constraint on the covariance matrix
                U, s, _ = np.linalg.svd(covNew)
                s[s<psi] = psi
                covNew = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
                starting_gmm = [(1.0, np.mean(DTR[:, LTR == 0], axis = 1), covNew)]
                all_gmm.append(starting_gmm)
            else:
                starting_gmm = all_gmm[0]
            new_gmm, _ = LBG(DTR[:, LTR == 0], starting_gmm, t, alpha, psi, diag0, tied0)
            all_gmm[0] = new_gmm
            logdens, _ = logpdf_GMM(DTE, new_gmm)
            S[0, :] = logdens
    for count1 in range(int(np.log2(components1))):
       
        # Train one GMM for each class
            if count1 == 0:
                # Start from max likelihood solution for one component
                covNew = np.cov(DTR[:, LTR == 1])
                # Impose the constraint on the covariance matrix
                U, s, _ = np.linalg.svd(covNew)
                s[s<psi] = psi
                covNew = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
                starting_gmm = [(1.0, np.mean(DTR[:, LTR == 1], axis = 1), covNew)]
                all_gmm.append(starting_gmm)
            else:
                starting_gmm = all_gmm[1]

            # Train the new components and compute log-densities
            new_gmm, _ = LBG(DTR[:, LTR == 1], starting_gmm, t, alpha, psi, diag1, tied1)
            all_gmm[1] = new_gmm
            logdens, _ = logpdf_GMM(DTE, new_gmm)
            S[1, :] = logdens

            # Compute minDCF for the model with the current number of components
    llr = compute_llr(S)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)


            # if f == 0:
            #     # print("Components0: %d, Components1: %d      min DCF: %f" % (2**(count0 + 1), 2**(count1 + 1), minDCF))
            # else:
            #     # Save results on file
            #     print("Components0: %d, Components1: %d      min DCF: %f" % (2**(count0 + 1), 2**(count1 + 1), minDCF))
            #     f.write("\ncomponents: " + str(2**(count0 + 1)) +", " +str(2**(count1 + 1)) + "\n")
            #     f.write("\n" + type + ": " + str(minDCF) + "\n")
   
    return llr, minDCF
















# Perform k-fold cross validation on test data for the specified model
def k_fold_cross_validation(D, L, k, pi, Cfp, Cfn, diag, tied, components, pca_dim=None, seed = 0):

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
        llr[idxTest], _ = GMM_classifier(DTR, LTR, DTE, LTE, 2, components, pi, Cfn, Cfp, diag, tied)
        start_index += elements

    # Evaluate results after all k-fold iterations (when all llr are available)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, llr







def k_fold_cross_validation_1(D, L, k, pi, Cfp, Cfn, components0, components1, diag0, tied0, diag1, tied1, pca_dim=None, seed = 0):

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
        llr[idxTest], _ = GMM_classifier_1(DTR, LTR, DTE, LTE, 2, components0, components1, diag0, tied0, diag1, tied1, pi, Cfn, Cfp, t = 1e-6, psi = 0.01, alpha = 0.1, f=0, type="")
        start_index += elements

    # Evaluate results after all k-fold iterations (when all llr are available)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, llr

def results_GMM(DTR, LTR, comp0, comp1, k, pi, Cfp, Cfn, diag0, tied0, diag1, tied1, pca_dim ):
      
      minDCF, _ = k_fold_cross_validation_1(DTR, LTR, k, pi, Cfp, Cfn, comp0, comp1, diag0, tied0, diag1, tied1, pca_dim)
      return minDCF









if __name__ == "__main__":

    ### Train and evaluate different GMM models using cross validation and single split
    ### Plot figures for hyperparameter optimization



    D, L = load("../Train.txt")
    DN = Z_score(D)
    components_val0 = [2, 4, 8, 16]
    components_val1 = [1, 2, 4]

    # for tied0 in [True, False]:
    #     for diag0 in [True, False]:
    #         for tied1 in [True, False]:
    #             for diag1 in [True, False]:
    #                 for i in components_val0:
    #                     for j in components_val1:
    #                         print(f"component {i}, {j},{diag0}, {tied0}, {diag1} {tied1} : ")
    #                         minDCF = results_GMM(D, L, i, j, 5, 1/11, 1, 1, diag0, tied0, diag1, tied1, pca_dim=None)
    #                         filename = "./Results/GMM/GMM_results.txt"
    #                         with open(filename, "w") as f:
    #                             f.write(f"\ncomponents ({i}, {j} {diag0} {tied0} {diag1} {tied1}): "  + "\n")
    #                         print(f"{minDCF} + '\n'")


    
    # k = 5
    # pi = 1/11
    # Cfn, Cfp = 1, 1
    # pca_dim = 6
    # filename = "./Results/GMM/GMM_results.txt"

    # DCF_z = np.zeros([4, len(components_val1)])
    
    # with open(filename, "w") as f:
    #     plt.figure()
    #     f.write("**** min DCF for different GMM models *****\n\n")
    #     for tied, diag, pca in product((True, False),(True, False),(None, pca_dim)):
    #         f.write("\nTied: " + str(tied) + "\n")
    #         f.write("\nDiag: " + str(diag) + "\n")
    #         f.write("\nPCA: " + str(pca) + "\n")
    #         DCF = []
    #         for components in components_val:
    #             f.write("\ncomponents: " + str(components) + "\n")
    #             minDCF, _ = k_fold_cross_validation(DN, L, k, pi, Cfp, Cfn, diag, tied, components, pca)
    #             DCF.append(minDCF)
    #             f.write("\nZ-norm: " + str(minDCF) + "\n")
    #         label = f"GGM-{'tied' if tied else 'not tied'}-{'diag' if diag else 'not diag'}-{'PCA6' if pca else ''}"
    #         plt.plot(components_val, DCF, marker='o', linestyle='dashed', label=label)
    # plt.xlabel("Components")
    # plt.ylabel("min DCF")
    # plt.legend()
    # plt.savefig("../Visualization/GMM.png")

     #Training the best GMM model on the working point 0.2
    minDCF = results_GMM(D, L, 8, 2, 5, 0.2, 1, 1, True, False, True, False, pca_dim=None)
    print(minDCF)                          

