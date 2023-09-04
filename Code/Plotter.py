import numpy
import matplotlib
import scipy
import matplotlib.pyplot as plt

import constants as cnst
import seaborn as sb

import MLCore as MLCore
import Metrics as metrics

def mCol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DataList = []
    LabelList = []
    with open(fname) as f:
        for line in f:
            try:
                attributes = line.split(',')[0:cnst.N_ATTR]
                attributes = mCol(numpy.array([float(i) for i in attributes]))
                label = int(line.split(',')[-1].strip())
                DataList.append(attributes)
                LabelList.append(label)
            except:
                pass

    return numpy.hstack(DataList), numpy.array(LabelList, dtype=numpy.int32)

def plot_hist_all_features(Data, Label, title):

    Data0 = Data[:, Label==0]
    Data1 = Data[:, Label==1]

    for dIdx in range(cnst.N_ATTR):
        plt.figure()
        plt.xlabel("Attribute " + str(dIdx))
        plt.hist(Data0[dIdx, :], bins = cnst.BINS, density = True, alpha = 0.4, label = 'spoofed fingerprint',   edgecolor = 'black', linewidth = 1.0)
        plt.hist(Data1[dIdx, :], bins = cnst.BINS, density = True, alpha = 0.4, label = 'authentic fingerprint', edgecolor = 'black', linewidth = 1.0)
        plt.legend(fontsize = "10")
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('Visualization/%s/hist%d.png' % (title, dIdx))
    plt.close('all')

def plot_scatter_all_features(D, L, title):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    for dIdx1 in range(cnst.N_ATTR):
        for dIdx2 in range(cnst.N_ATTR):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel('Attribute %d' %dIdx1)
            plt.ylabel('Attribute %d' %dIdx2)
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'spoofed fingerprint')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'authentic fingerprint')
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('Visualization/Scatter/scatter_%d_%d.pdf' % (title, dIdx1, dIdx2))
        plt.close('all')

    
def heatmap(X, label: str = ""):
    plt.figure()
    sb.heatmap(numpy.corrcoef(X))
    plt.savefig(f'Visualization/Heatmaps/{label}.png')



"""

Plotting PCA

"""
import numpy 
import matplotlib.pyplot as plt

def vcol(m):
    """Reshape as a column vector input m"""
    return m.reshape((m.size,1))



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
    plt.savefig('Visualization/Scatter/PCA_2.png', dpi = 200)
    plt.show()
    
    
def PCA(D):
    N = D.shape[1]
    mu = vcol(D.mean(1)) # compute mean by column of the dataset for each dimension, note that mu is a row vector of shape (4,)
    DC = D - mu # center data
    C = numpy.dot(DC, DC.T)/N # compute the covariance matrix of centered data
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:2] # 2 dim PCA
    DP = numpy.dot(P.T, D)
    return DP


"""
"""



"""

Plot LDA

"""


def LDA(D,L,directions = 1):
    "Perform LDA on a dataset D where each column is a sample and each row a feature"
    NC = 2 # number of classes in the dataset
    N = D.shape[1] # number of samples in dataset
    mu = mCol((D.mean(1))) 
    samples = [] # will contain the samples of i-th class (0,1,2) in the i-th position
    muC = [] # will contain the mean of i-th class in the i-th position as column vector
    for i in range(NC):
        samples.append(D[: , L==i])  # get samples for each class
        muC.append(mCol(D[: , L==i].mean(1))) #compute mean for each class
                   
    #compute SW
    SWC = [] # will contain SW,c for each class where SW,c is the covariance matrix of each class
    for i in range(NC):
        #CC is for Centered Class, the samples of a specific class centered (subtractaction of class mean from all class samples)
        CC = samples[i] - muC[i]
        SWC.append(numpy.dot(CC,CC.T)/samples[i].shape[1]) #compute SW for the i-th class
    
    s=0 # will contain sum of (SW,c * Nc) where Nc is the number of samples for a class
    for i in range(NC):
        s += SWC[i] * samples[i].shape[1]
        SW = s/N # compute the SW matrix
    #compute SB
    summation=0
    for i in range(NC):
        temp = muC[i] - mu
        summation += numpy.dot(temp, temp.T) * samples[i].shape[1]
    SB = summation/N
    # solve the Generalized eigenvalue problem
    m = directions 
    s, U = scipy.linalg.eigh(SB,SW)
    W = U[:, ::-1][:,0:m] # matrix W is the one that makes possible to perform LDA projecting samples in the new space
    return numpy.dot(W.T, D) # dataset after LDA
    

def plot_hist(D, L, title):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    
    plt.figure()
    plt.hist(D0[0,:], bins = cnst.BINS, alpha = 0.8, label = 'spoofed fingerprint ')
    plt.hist(D1[0,:], bins = cnst.BINS, alpha = 1, label = 'authentic fingerprint')
    plt.yscale("linear")
    plt.legend()
    
    plt.savefig('Visualization/Histograms/LDA.png')
    #plt.title(title)
    plt.show()



"""

Plot : Variance function of number of dimensions

"""
def plot_variance_pca(DTR, LTR):
    DTR, eigValue = MLCore.PCA_projected(DTR, DTR.shape[0] + 1)
    n_dimensions = eigValue.size
    sorted_eigenvalues = eigValue[::-1]
    total_variance = numpy.sum(sorted_eigenvalues)
    explained_variance_ratio = numpy.cumsum(sorted_eigenvalues / total_variance)
    plt.figure()
    ax = plt.axes()
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.plot(range(1, n_dimensions + 1), explained_variance_ratio)
    plt.xlabel('PCA dimensions')
    plt.ylabel('Fraction of explained variance')
    plt.savefig("Visualization/explained_variance/explained_variance.png")
    plt.close()


def plot_bayess_error(llr, label):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    # for each value of p we compute the effective prior
    plt.figure()
    llr = llr.ravel()
    dcf = numpy.zeros(21)
    min_dcf  = numpy.zeros(21)

    for(idx, p) in enumerate(effPriorLogOdds):
        pi = 1 / ( 1 + numpy.exp(-p))
        dcf[idx] = metrics.DCF_norm(llr, label, pi, 1, 1)
        min_dcf[idx] = metrics.minimun_DCF_norm(llr, label, pi, 1, 1)[0]

    plt.plot(effPriorLogOdds, dcf, label='actDCF', color='b')
    plt.plot(effPriorLogOdds, min_dcf, label="minDCF", color='r')
    plt.ylim([0, 1])
    plt.xlim([-3, 3])
    plt.show()

def plot_roc(llr, label):
   
    tpr, fpr = metrics.roc_points(llr, label)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # Change default font size 
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    D, L = load('Train.txt')
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    # plot_hist_all_features(D, L)
    # #plot_scatter_all_features(D, L)

    #features correlation using heatmap

    


    # Data Analysis
    # D_PCA = PCA(D)
    # plot_scatter(D_PCA, L, "PCA")
    # D_LDA = LDA(D,L)
    # plot_hist(D_LDA, L, 'Histogram of dataset features - LDA direction')
    heatmap(D, "heatmap_rawData")
    heatmap(D[:, L == 0], "heatmap_label0_rawData")
    heatmap(D[:, L == 1], "heatmap_label1_rawData")

    plot_variance_pca(D, L)
    print(D0.shape)
    print(D1.shape)

