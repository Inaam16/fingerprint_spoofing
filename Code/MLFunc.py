import numpy
import MLCore as MLCore
priorProbability = 0.5

def kfoldRes(function):
    Data = numpy.load('Datasets/mainData.npy', allow_pickle=True)
    Label = numpy.load('Datasets/mainLabels.npy', allow_pickle=True)

    PCAaccuracy = applyPrePro(Data, Label, priorProbability, 'PCA', function)
    idxPCA = numpy.argmax(PCAaccuracy)
    bestPCA = PCAaccuracy[idxPCA]
    print(f'PCA {idxPCA+1}: {bestPCA*100:.02f}%')

    LDAaccuracy = applyPrePro(Data, Label, priorProbability, 'LDA', function)
    idxLDA = numpy.argmax(LDAaccuracy)
    bestLDA = LDAaccuracy[idxLDA]
    print(f'LDA {idxLDA+1}: {bestLDA*100:.02f}%')
    
    PCALDAaccuracy = applyPrePro2D(Data, Label, priorProbability, 'PCA', 'LDA', function)
    idxPCALDA = numpy.unravel_index(numpy.argmax(PCALDAaccuracy),PCALDAaccuracy.shape)
    bestPCALDA = PCALDAaccuracy[idxPCALDA]
    print(f'PCA {idxPCALDA[1]+1}, LDA {idxPCALDA[0]+1}: {bestPCALDA*100:.02f}%')

    LDAPCAaccuracy = applyPrePro2D(Data, Label, priorProbability, 'LDA', 'PCA', function)
    idxLDAPCA = numpy.unravel_index(numpy.argmax(LDAPCAaccuracy), LDAPCAaccuracy.shape)
    bestLDAPCA = LDAPCAaccuracy[idxLDAPCA]
    print(f'LDA {idxLDAPCA[0]+1}, PCA {idxLDAPCA[1]+1}: {bestLDAPCA*100:.02f}%')


def z_score(D):
    #We neeed mean of each feature
    meanFeatures = numpy.mean(D,0)
    #standard deviation of each feature
    standardFeatures = numpy.std(D,0)

    processedD = D - meanFeatures / standardFeatures
    return meanFeatures, standardFeatures, processedD


def applyPrePro(Data, Label, priorProbability, name, function):
    parentAccuracy = MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], function)
    accuracy = [parentAccuracy]
    for dim in range(2, Data.shape[0]+1):
        Data = numpy.load(f'Datasets/{name}{dim:02d}.npy', allow_pickle=True)
        accuracy.append(MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], function))
    return numpy.array(accuracy)

def applyPrePro2D(Data, Label, priorProbability, name1, name2, function):
    parentAccuracy = MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], function)
    accuracy=numpy.zeros((Data.shape[0], Data.shape[0]))
    accuracy[0,0] += parentAccuracy
    for dim1 in range(2, Data.shape[0]+1):
        for dim2 in range(2, dim1+1):
            Data = numpy.load(f'Datasets/{name1}{dim1:02d}+{name2}{dim2:02d}.npy', allow_pickle=True)
            accuracy[dim1-1, dim2-1] += MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], function)
    return accuracy


if __name__ == '__main__':
    Data = numpy.load('Datasets/mainData.npy', allow_pickle=True)
    Label = numpy.load('Datasets/mainLabels.npy', allow_pickle=True)
    priorProbability = 0.5
    print('MVG:')
    kfoldRes(MLCore.MVG)
    print('NB:')
    kfoldRes(MLCore.naiveBayesGC)
    print('TMVG:')
    kfoldRes(MLCore.tied_MVG)
    print('TNB:')
    kfoldRes(MLCore.tied_naiveBayesGC)
    print('LR:')
    kfoldRes(MLCore.LR)

    (DTR, LTR), (DTE, LTE) = MLCore.DBSplit(Data, Label, 2 , 3, seed=0)

    acc = MLCore.MVG(DTR, LTR, DTE, LTE, [0.5, 0.5]).mean()*100
    print('acc is')
    print(acc)


    
    # MVGA = MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.MVG)
    # print(f'MVG: {MVGA*100:0.1f}%')

    # MVGaccuracy = []
    # for LDAdim in range(2, Data.shape[0]):
    #     Data = numpy.load(f'Datasets/PCA{LDAdim:02d}.npy', allow_pickle=True)
    #     MVGaccuracy.append((LDAdim,MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.MVG)))
    # for acc in MVGaccuracy:
    #     if (acc[1] >= MVGA):
    #         print(f'PCA {acc[0]}: {acc[1]*100:0.1f}') 
    
    # MVGaccuracy = []
    # for LDAdim in range(2, Data.shape[0]):
    #     Data = numpy.load(f'Datasets/LDA{LDAdim:02d}.npy', allow_pickle=True)
    #     MVGaccuracy.append((LDAdim,MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.MVG)))
    # for acc in MVGaccuracy:
    #     if (acc[1] >= MVGA):
    #         print(f'LDA {acc[0]}: {acc[1]*100:0.1f}')

    # NBA = MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.naiveBayesGC)
    # print(f'naive Bayes: {NBA*100:0.1f}%')
    # NBaccuracy = []
    # for LDAdim in range(2, Data.shape[0]):
    #     Data = numpy.load(f'Datasets/PCA{LDAdim:02d}.npy', allow_pickle=True)
    #     NBaccuracy.append((LDAdim,MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.naiveBayesGC)))
    # for acc in NBaccuracy:
    #     if (acc[1] >= NBA):
    #         print(f'PCA {acc[0]}: {acc[1]*100:0.1f}')

    # MVGaccuracy = []
    # for LDAdim in range(2, Data.shape[0]):
    #     Data = numpy.load(f'Datasets/LDA{LDAdim:02d}.npy', allow_pickle=True)
    #     MVGaccuracy.append((LDAdim,MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.naiveBayesGC)))
    # for acc in MVGaccuracy:
    #     if (acc[1] >= MVGA):
    #         print(f'LDA {acc[0]}: {acc[1]*100:0.1f}')

    # print(f'tied MVG: {MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.tied_MVG)*100:0.1f}%')
    # print(f'tied Naive Bayes: {MLCore.k_fold(Data, Label, [priorProbability, 1-priorProbability], MLCore.tied_naiveBayesGC)*100:0.1f}%')