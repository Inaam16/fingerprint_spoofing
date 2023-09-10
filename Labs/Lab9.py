#Lab9 - SVM

import numpy
import scipy


import sklearn.datasets
#Loading the dataset
def load_iris():
    D,L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D,L

#split the dataset into  2 parts -> training and validation sets
def split_db_2to1(D, L, seed = 0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    #permutation receives an array and compute random permutations
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)
#Applying the functions on our dataset
D, L = load_iris()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D,L)



# reshape a numpy row into a column
def mCol(row):
    return row.reshape((row.size, 1))
# reshape a numpy column into a row
def mRow(col):
    return col.reshape((1, col.size))


"""

We have the primal and dual formulation

for the primal we look for the minimizer of (w,b)

for the dual we look for maximizer of alpha


For linear SVM we can optimizer either the dual or the primal
UNFORTUANITLY, the primal objective function is non-differentiable


"""

"""
sum(alphai zi) = 0 constraint cannot be covered by LBFGS so we change our function to cover
this constraint

1st part of the problem: solve the linear formulation





"""
# solving for linear model

def obj_svm_wrapper(H):
    def lagrangian(alpha):
        alpha = mCol(alpha)
        grad = mRow(H.dot(alpha) - numpy.ones((alpha.shape[0], 1))) 
        obj_l = 0.5*alpha.T.dot(H).dot(alpha) - alpha.T @ numpy.ones(alpha.shape[0])
        return obj_l, grad
    return lagrangian


def compute_weights(C, LTR, pi):
    bounds = numpy.zeros((LTR.shape[0]))
    empirical_pi = (LTR == 1).sum() / LTR.shape[0]
    bounds[LTR == 1] = C * pi / empirical_pi
    bounds[LTR == 0] = C * (1-pi) / (1 - empirical_pi)
    return list(zip(numpy.zeros(LTR.shape[0]), bounds))


def compute_primal_obj(w, C, Z, D):
    w = mCol(w)
    z = mRow(Z)
    param1 = 0.5 * (w*w).sum()
    param2 = z*numpy.dot(w.T, D)
    param3 = 1 - param2
    val = numpy.maximum(numpy.zeros(param3.shape), param3)
    param4 = numpy.sum(val)
    param5 = C*param4
    lossPr = param1 + param5
    return lossPr

def linear_SVM(DTR, LTR, DTE, K, C, pi):
    D = numpy.vstack([DTR, numpy.ones(DTR.shape[1])*K])
    G = numpy.dot(D.T, D)
    z = 2*LTR -1 #making the labels 1 or -1
    H = z @ z.T * G
    lag = obj_svm_wrapper(H)
    alpha, dual_obj, _ = scipy.optimize.fmin_l_bfgs_b(
        lag, 
        numpy.zeros(DTR.shape[1]), 
        bounds = [(0, C) for _ in range(DTR.shape[1])],
        factr = 1.0
    )
    w = numpy.dot(D, mCol(alpha) * mCol(z))
    DTE = numpy.vstack([DTE, numpy.ones(DTE.shape[1])*K])
    scores = numpy.dot(w.T, DTE)

    primal_loss = compute_primal_obj(w, C, z, D)
    dual_loss = -dual_obj
    dual_gap = primal_loss - dual_loss
    
    return primal_loss, dual_loss, scores, alpha



"""

Kernel SVM

SVM allows for non-linear classification through an "implicit" 
expansion of the feature space into a higher dimensional space

the SVM dual objective depends on the trainning samples only through dot products
=>
so for SVM it is sufficient that we are able to compute the scalar product
between the expaned features

k(x1, x2) = phi(x1).T * phi(x2)

k: kernel function

"""
"""
same code as before we just replace Hij = zizjxixj
with Hij = zizjk(xi,xj)

"""


def kernel_SVM(DTR, LTR, DTE, LTE, K, C, d, c):
    z = LTR*2 - 1
    k_train = ((numpy.dot(DTR.T, DTR) + c) **d) + K**2
    H = mCol(z)*mRow(z)*k_train
    lag = obj_svm_wrapper(H)
    alpha, dual_obj, _ = scipy.optimize.fmin_l_bfgs_b(
        lag,
        numpy.zeros(DTR.shape[1]),
        bounds = [(0, C) for _ in range(DTR.shape[1])],
        factr = 1.0
    )
    scores = (mCol(alpha)*mCol(z)*((DTR.T.dot(DTE) + c)**d + K**2)).sum(0)
    return scores, dual_obj

if __name__ == "__main__":
    
   primal_loss,dual_loss, scores, alhpa = linear_SVM(DTR, LTR, DTE, 1, 0.1, 0.5)
#    print(primal_loss)
#    print(dual_loss)
#    print(scores.shape)
#    C = 0.1
#    print([(0, C) for _ in range(DTR.shape[1])])
   scores_d, dual_loss_d = kernel_SVM(DTR, LTR, DTE, LTE, 1, 1, 2, 0)
   print(dual_loss_d)





