# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 18:48:51 2023

@author: maria
"""

# GMM LAB10
# ==== ====
"""
GMM

weighted sum of N gaussians

logpdf_GMM is a function that computes the density of a GMM for a set of samples

gmm = [{w1, mu1, C1}, {w2, mu2, C2}, ...]


"""
# Data = np.load("C:/Users/inaam/Downloads/GMM_data_4D.npy", allow_pickle=True)
# Label = np.load(
#     "C:/Users/inaam/Downloads/commedia_labels_infpar.npy", allow_pickle=True
# )


"""


We compute a matrix S
for the cell (1,1):
      => we have log(N(x1|mu1, sigma1)) + log(w1)
for the cell (1,2):
      => we have log(N(x1|mu2, sigma2)) + log(w2)


"""


def logpdf_GMM(X, gmm):
    # S is a matrix of number of rows = number of classes
    # number of columns = number of samples
    S = np.zeros((len(gmm), X.shape[1]))
    for i in range(X.shape[1]):
        for idx, component in enumerate(gmm):
            S[idx, i] = logPDF_Gau_ND(X[:, i : i + 1], component[1], component[2]) + np.log(
                component[0]
            )
    return S, scipy.special.logsumexp(S, axis=0)


def EM(X, gmm, psi):
    """EM algorithm has 2 steps:
    - computing the responsibilities for each component for each sample
    - updating the model parameters"""
    limit = 1e-6
    loss_new = None
    loss_old = None

    while loss_old is None or loss_new - loss_old > limit:
        loss_old = loss_new
        S_j = np.zeros((len(gmm), X.shape[1]))
        for idx in range(len(gmm)):
            S_j[idx, :] = logPDF_Gau_ND(X, gmm[idx][1], gmm[idx][2]) + np.log(gmm[idx][0])
        S_m = vrow(scipy.special.logsumexp(S_j, axis=0))

        # S_j, S_m = logpdf_GMM(X, gmm)
        S_p = np.exp(S_j - S_m)
        loss_new = np.mean(S_m)
        # M-Step
        Z = np.sum(S_p, axis=1)

        F = np.zeros((X.shape[0], len(gmm)))
        for idx in range(len(gmm)):
            F[:, idx] = np.sum(S_p[idx, :] * X, axis=1)

        S = np.zeros((X.shape[0], X.shape[0], len(gmm)))
        for idx in range(len(gmm)):
            S[:, :, idx] = np.dot(S_p[idx, :] * X, X.T)

        mu_new = F / Z
        C_new = S / Z

        for idx in range(len(gmm)):
            C_new[:, :, idx] -= np.dot(vcol(mu_new[:, idx]), vrow(mu_new[:, idx]))

        w_new = Z / np.sum(Z)

        gmm_new = [
            ((w_new[idx]), vcol(mu_new[:, idx]), C_new[:, :, idx]) for idx in range(len(gmm))
        ]

        for i in range(len(gmm_new)):
            C_new = gmm_new[i][2]
            u, s, _ = np.linalg.svd(C_new)
            s[s < psi] = psi
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], np.dot(u, vcol(s) * u.T))
        gmm = gmm_new
        # print(loss_new)
    return gmm_new


def LBG(X, gmm, n, alpha, psi):
    for i in range(len(gmm_init)):
        C_new = gmm_init[i][2]
        u, s, _ = np.linalg.svd(C_new)
        s[s < psi] = psi
        gmm_init[i] = (gmm_init[i][0], gmm_init[i][1], np.dot(u, vcol(s) * u.T))

    gmm_init = EM(X, gmm, psi)

    for i in range(n):
        print(i)
        print(gmm_init)
        gmm_new = []
        for g in range(len(gmm_init)):
            w_new = gmm_init[g][0] / 2
            C_g = gmm_init[g][2]
            u, s, _ = np.linalg.svd(C_g)
            d = u[:, 0:1] * s[0] ** 0.5 * alpha
            gmm_new.append((w_new, gmm_init[g][1] + d, C_g))
            gmm_new.append((w_new, gmm_init[g][1] - d, C_g))
        gmm_init = EM(X, gmm_new)
    return gmm_init


if __name__ == "__main__":
    gmm_init = [
        [0.3333333333333333, [[-2.0]], [[1.0]]],
        [0.3333333333333333, [[0.0]], [[1.0]]],
        [0.3333333333333333, [[2.0]], [[1.0]]],
    ]
    # gmm_final = EM(Data, gmm_init)
    # print(gmm_final)
    gmm_final = LBG(Data, gmm_init, 1, 0.1)
    print(gmm_final)
