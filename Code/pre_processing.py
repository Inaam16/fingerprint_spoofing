from MLCore import *


def project_PCA(Data, eigen_vectors):
    DTR = np.dot(eigen_vectors.T, Data)
    return DTR


def PCA(Data, dim, *, components=False):
    """Return a new dataset projected in the new subspace
    NEW: addded the dimEigVect as returned value"""
    eigen_vectors, eigen_values = principal_components(Data, dim)
    DTR = np.dot(eigen_vectors.T, Data)
    if components:
        return DTR, eigen_vectors, eigen_values
    return DTR





# def LDA(Data, Label, dim):
#     """Apply Linear Discriminant to the Data and returns the #dim most important dimensions"""
#     Sw, Sb = SwSb(Data, Label)
#     eigVal, eigVect = scipy.linalg.eigh(Sb, Sw)
#     dimeigVect = eigVect[:, ::-1][:, 0:dim]
#     return dimeigVect


# def LDA_projected(Data, Label, dim):
#     dimeigVect = LDA(Data, Label, dim)
#     DTR = np.dot(dimeigVect.T, Data)
#     return DTR, dimeigVect


def principal_components(Data, dim):
    """Apply Principal Component Analysis to the Data and return the dim most important dimensions"""
    mu, cov = mean_and_covariance(Data)
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    dim_eigen_vects = eigen_vectors[:, ::-1][:, 0:dim]
    dim_eigen_vals = eigen_values[::-1][0:dim]
    return dim_eigen_vects, dim_eigen_vals


# def SwSb(Data, Label):
#     Sw = 0
#     Sb = 0
#     mu = mCol(Data.mean(1))
#     for id in range(Label.max() + 1):
#         lData = Data[:, Label == id]
#         lmu = mCol(lData.mean(1))
#         Sb += lData.shape[1] * np.dot(lmu - mu, (lmu - mu).T)
#         Sw += np.dot(lData - lmu, (lData - lmu).T)
#     Sb /= Data.shape[1]
#     Sw /= Data.shape[1]
#     return Sw, Sb