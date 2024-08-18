import numpy as np
import scipy
import matplotlib.pyplot as plt

import constants as cnst
import seaborn as sb

# import MLCore as MLCore
import Metrics as metrics
from Utilities import *
from pre_processing import PCA




#Plotting the histograms  of all features 
def plot_hist_all_features(Data, Label, title):
    Data0 = Data[:, Label == 0]
    Data1 = Data[:, Label == 1]

    for dIdx in range(cnst.N_ATTR):
        plt.figure()
        plt.xlabel("Attribute " + str(dIdx))
        plt.hist(
            Data0[dIdx, :],
            bins=cnst.BINS,
            density=True,
            alpha=0.4,
            label="spoofed fingerprint",
            edgecolor="black",
            linewidth=1.0,
        )
        plt.hist(
            Data1[dIdx, :],
            bins=cnst.BINS,
            density=True,
            alpha=0.4,
            label="authentic fingerprint",
            edgecolor="black",
            linewidth=1.0,
        )
        plt.legend(fontsize="10")
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig(f"./Visualization/Histograms/{title}-hist{dIdx}.png")
    plt.close("all")



#Drawing scatter plots of features two by two and histograms for each feature
def draw_scatterplots(D, L):
    # Plots of the different attribute pairs for each class
    fig, axes = plt.subplots(
        nrows=cnst.N_ATTR, ncols=cnst.N_ATTR, figsize=(3 * cnst.N_ATTR, 3 * cnst.N_ATTR)
    )

    for i, axs in enumerate(axes):
        for j, ax in enumerate(axs):
            if i == j:  # Draw histogram
                for c, name in enumerate(cnst.CLASS_NAMES):
                    mask = L == c  # Mask to select each class and draw its histogram
                    ax.hist(D[i, mask], bins=20, density=True, alpha=0.5)
            else:  # Draw scatterplot
                for c, name in enumerate(cnst.CLASS_NAMES):
                    ax.scatter(D[i, L == c], D[j, L == c], alpha=0.5)
    plt.savefig(f"./Visualization/Scatter/scatterplots", bbox_inches="tight")
    plt.close()

#Heat map function for correlations
def heatmap(X, label: str = ""):
    plt.figure()
    sb.heatmap(np.corrcoef(X))
    plt.savefig(f"./Visualization/Heatmaps/Heatmaps-{label}.png")
    plt.close()



#Linear discriminant analysis
def LDA(D, L, directions=1):
    "Perform LDA on a dataset D where each column is a sample and each row a feature then plot it"
    NC = 2  # number of classes in the dataset
    N = D.shape[1]  # number of samples in dataset
    mu = mCol((D.mean(1)))
    samples = []  # will contain the samples of i-th class (0,1,2) in the i-th position
    muC = []  # will contain the mean of i-th class in the i-th position as column vector
    for i in range(NC):
        samples.append(D[:, L == i])  # get samples for each class
        muC.append(mCol(D[:, L == i].mean(1)))  # compute mean for each class

    # compute SW
    SWC = []  # will contain SW,c for each class where SW,c is the covariance matrix of each class
    for i in range(NC):
        # CC is for Centered Class, the samples of a specific class centered (subtractaction of class mean from all class samples)
        CC = samples[i] - muC[i]
        SWC.append(np.dot(CC, CC.T) / samples[i].shape[1])  # compute SW for the i-th class

    s = 0  # will contain sum of (SW,c * Nc) where Nc is the number of samples for a class
    for i in range(NC):
        s += SWC[i] * samples[i].shape[1]
        SW = s / N  # compute the SW matrix
    # compute SB
    summation = 0
    for i in range(NC):
        temp = muC[i] - mu
        summation += np.dot(temp, temp.T) * samples[i].shape[1]
    SB = summation / N
    # solve the Generalized eigenvalue problem
    m = directions
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][
        :, 0:m
    ]  # matrix W is the one that makes possible to perform LDA projecting samples in the new space
    return np.dot(W.T, D)  # dataset after LDA

#Two Plots LDA 
def plot_hist(D, L, title):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.figure()
    plt.hist(D0[0, :], bins=cnst.BINS, alpha=0.8, label="spoofed fingerprint ")
    plt.hist(D1[0, :], bins=cnst.BINS, alpha=0.8, label="authentic fingerprint")
    plt.yscale("linear")
    plt.legend()
    plt.title(title)
    plt.savefig("./Visualization/Histograms/Histograms-LDA.png")
    plt.show()
    plt.close()

#Explained variance PCA
def plot_variance_pca(DTR, LTR):
    """Plot : Variance function of number of dimensions"""
    n_dimensions = DTR.shape[0]
    _, eigen_values = MLCore.principal_components(DTR, n_dimensions)
    total_variance = np.sum(eigen_values)
    explained_variance_ratio = np.cumsum(eigen_values / total_variance)
    plt.figure()
    ax = plt.axes()
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.xaxis.grid(color="gray", linestyle="dashed")
    plt.plot(range(1, n_dimensions + 1), explained_variance_ratio)
    plt.xlabel("PCA dimensions")
    plt.ylabel("Fraction of explained variance")
    plt.savefig("./Visualization/Scatter/explained_variance.png")
    plt.close()

#Two dimensional PCA and plotting it
# def PCA(D):
#     N = D.shape[1]
#     mu = vcol(D.mean(1)) # compute mean by column of the dataset for each dimension, note that mu is a row vector of shape (4,)
#     DC = D - mu # center data
#     C = np.dot(DC, DC.T)/N # compute the covariance matrix of centered data
#     s, U = np.linalg.eigh(C)
#     P = U[:, ::-1][:, 0:2] # 2 dim PCA
#     DP = np.dot(P.T, D)
#     return DP


def plot_scatter(D, L, title):
    
    # make data that will be plotted
    label1 = (L == 1)
    label0 = (L == 0)
    
    Data_0 = D[:, label0]
    Data_1 = D[:, label1]

    plt.figure()
    plt.scatter(Data_0[0, :], Data_0[1, :], label='spoofed fingerprint', s=16, alpha = 0.5)
    plt.scatter(Data_1[0, :], Data_1[1, :], label='authentic fingerprint',s=16, alpha = 0.5)

    
    #plt.title(title)
    plt.legend()
    plt.savefig('./Visualization/Scatter/PCA_2.png', dpi = 200)
    plt.show()
    
    
if __name__ == "__main__":
    # Change default font size
    plt.rc("font", size=16)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    D, L = load("./Train.txt")
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    plot_hist_all_features(D, L, 'Histogram of all features')
   

    # features correlation using heatmap
    # Data Analysis
    D_PCA = PCA(D,2)
    plot_scatter(D_PCA, L, "PCA")
    D_LDA = LDA(D,L)
    plot_hist(D_LDA, L, 'Histogram of dataset features - LDA direction')
    heatmap(D, "heatmap_rawData")
    heatmap(D[:, L == 0], "heatmap_label0_rawData")
    heatmap(D[:, L == 1], "heatmap_label1_rawData")

    draw_scatterplots(D, L)

    plot_variance_pca(D, L)
    print(D0.shape)
    print(D1.shape)


