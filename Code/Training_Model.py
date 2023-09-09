import numpy
import MLCore as MLCore
import MLFunc as MLFunc
import Metrics
import Models as models
import Plotter as Plotter


def train_gaussian(DTR, LTR, Model, k, pca, pca_m = None, lda = False, lda_m = None, pi = 0.5, Cfn = 1, Cfp = 1, znorm = False):
    n_samples = DTR.shape[1]
    foldSize = n_samples // k
    numpy.random.seed(seed=0)
    idx = numpy.random.permutation(DTR.shape[1])

    scores = numpy.zeros(LTR.shape[0])
    labels = numpy.zeros(LTR.shape[0])

    for i in range(k):
        foldIdx = (i * foldSize, (i + 1) * foldSize)
        testIdx = idx[foldIdx[0]:foldIdx[1]]
        trainIdx = numpy.concatenate([idx[:foldIdx[0]], idx[foldIdx[1]:]])
        trainData, trainLabel = DTR[:, trainIdx], LTR[trainIdx]
        evaluationData, evaluationLabel = DTR[:, testIdx], LTR[testIdx]

    if znorm == True:
        _, _, trainData = MLFunc.z_score(trainData)
        _, _, evaluationData = MLFunc.z_score(evaluationData)

    if pca == True:
        trainData, P, _ = MLCore.PCA_projected(trainData, pca_m)
        evaluationData = numpy.dot(P.T, evaluationData)

    if lda == True:
         trainData, P = MLCore.LDA_projected(trainData, lda_m)
         evaluationData, P = numpy.dot(P.T, evaluationData) 

    if Model == "MVG":
        means, cov_matrices = models.compute_mean_cov_classes(trainData, trainLabel)
        llr = models.compuet_gaussian_scores(evaluationData, means, cov_matrices)
        scores[i * foldSize: (i + 1) * foldSize] = llr
        labels[i * foldSize: (i + 1) * foldSize] = evaluationLabel
    minDCF = Metrics.minimun_DCF_norm(scores, labels, pi, Cfn, Cfp)

    if Model == "NVB":
            means, cov_matrices = models.compute_mean_cov_classes(trainData, trainLabel)
            llr = models.train_naiveBayesGC(evaluationData, means, cov_matrices)
            scores[i * foldSize: (i + 1) * foldSize] = llr
            labels[i * foldSize: (i + 1) * foldSize] = evaluationLabel
    minDCF = Metrics.minimun_DCF_norm(scores, labels, pi, Cfn, Cfp)


    if Model == "TMVG":
            means, cov_matrices = models.compute_mean_cov_classes(trainData, trainLabel)
            llr = models.train_tied_MVG(trainData, trainLabel, evaluationData, means, cov_matrices)
            scores[i * foldSize: (i + 1) * foldSize] = llr
            labels[i * foldSize: (i + 1) * foldSize] = evaluationLabel
    minDCF = Metrics.minimun_DCF_norm(scores, labels, pi, Cfn, Cfp)


    if Model == "TNVB":
            means, cov_matrices = models.compute_mean_cov_classes(trainData, trainLabel)
            llr = models.train_tied_naiveBayesGC(trainData, trainLabel, evaluationData, means, cov_matrices)
            scores[i * foldSize: (i + 1) * foldSize] = llr
            labels[i * foldSize: (i + 1) * foldSize] = evaluationLabel
    minDCF = Metrics.minimun_DCF_norm(scores, labels, pi, Cfn, Cfp)
    return minDCF



if __name__ == "__main__":
    D, L = Plotter.load('Train.txt')
    minDCF,_  = train_gaussian(D, L, "MVG", k=5, pca=False, pca_m=9, pi=0.5, Cfn=1, Cfp=10, znorm=True)
    print("minDCF")
    print(minDCF)
