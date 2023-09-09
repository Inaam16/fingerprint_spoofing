import numpy as np
import constants as cnst


def mCol(row):
    """Reshape a numpy row into a column"""
    return row.reshape((row.size, 1))


def mRow(col):
    """Reshape a numpy column into a row"""
    return col.reshape((1, col.size))


def vcol(vector):
    return vector.reshape((vector.shape[0], 1))


def vrow(vector):
    return vector.reshape((1, vector.shape[0]))


def load(filename):
    DataList, LabelList = list(), list()
    with open(filename) as file:
        for line in file:
            attributes = line.split(",")[0 : cnst.N_ATTR]
            attributes = mCol(np.array(attributes, dtype=np.float64))
            label = int(line.split(",")[-1].strip())
            DataList.append(attributes)
            LabelList.append(label)

    return np.hstack(DataList), np.array(LabelList, dtype=np.int32)
