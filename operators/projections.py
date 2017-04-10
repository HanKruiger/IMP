from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import numpy as np


def mds_projection(source, **parameters):
    mds = MDS(**parameters)

    X = source.data()
    Y = mds.fit_transform(X)

    dataset = Dataset(Y, source.indices(), name='MDS projection')
    return dataset


def lamp_projection(source, landmarks_nd, landmarks_2d):
    X_s = landmarks_nd.data()
    Y_s = landmarks_2d.data()
    X = source.data()

    N = X.shape[0]
    n = Y_s.shape[1]

    Y = np.zeros((N, n))

    for i in np.arange(N):
        x = X[i, :]

        alphas = 1 / ((X_s - x)**2).sum(axis=1)

        x_tilde = alphas.dot(X_s) / alphas.sum()
        y_tilde = alphas.dot(Y_s) / alphas.sum()

        alphas_T = alphas[:, np.newaxis]
        A = alphas_T**0.5 * (X_s - x_tilde)
        B = alphas_T**0.5 * (Y_s - y_tilde)

        U, _, V = np.linalg.svd(A.T.dot(B), full_matrices=False)

        Y[i, :] = (x - x_tilde).dot(U.dot(V)) + y_tilde

    dataset = Dataset(Y, source.indices(), name='LAMP projection')

    return dataset
