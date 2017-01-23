from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np
from sklearn.decomposition import PCA


# Normalize in [0, 1] x [0, 1] (without changing aspect ratio)
def normalize(Y):
    Y_cpy = Y.copy()
    # Translate s.t. smallest values for both x and y are 0.
    for dim in range(Y.shape[1]):
        Y_cpy[:, dim] += -Y_cpy[:, dim].min()
        
    # Scale s.t. max(max(x, y)) = 1 (while keeping the same aspect ratio!)
    scaling = 1 / (np.absolute(Y_cpy).max())
    Y_cpy *= scaling

    # Translate s.t. the 'smaller' dimension is centralized.
    if Y_cpy[:, 0].max() < Y_cpy[:, 1].max():
        Y_cpy[:, 0] += 0.5 * (1 - Y_cpy[:, 0].max())
    else:
        Y_cpy[:, 1] += 0.5 * (1 - Y_cpy[:, 1].max())

    return Y_cpy

class PCAEmbedder(QThread):
    embedding_finished = pyqtSignal(np.ndarray)

    def __init__(self, X, n_components=2):
        super().__init__()
        self.X = X
        self.n_components = n_components

    def run(self):
        pca = PCA(n_components=self.n_components)
        Y = pca.fit_transform(self.X)
        Y = normalize(Y)
        self.embedding_finished.emit(Y)

