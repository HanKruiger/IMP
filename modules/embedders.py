from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Embedder(QThread):

    def __init__(self, n_components=2):
        super().__init__()
        self.n_components = n_components

    def set_input(self, X):
        self.X = X

    @abc.abstractmethod
    def run(self):
        """Method that should run the embedding"""

    @staticmethod
    def normalize(Y):
        Y_cpy = Y.copy()
        # Translate s.t. smallest values for both x and y are 0.
        for dim in range(Y.shape[1]):
            Y_cpy[:, dim] += -Y_cpy[:, dim].min()
            
        # Scale s.t. max(max(x, y)) = 1 (while keeping the same aspect ratio!)
        scaling = 1 / (np.absolute(Y_cpy).max())
        Y_cpy *= scaling

        # Centralize the median
        Y_cpy -= np.median(Y_cpy, axis=0)

        return Y_cpy

    @classmethod
    def parameters(cls):
        return [
            ('n_components', int, 2)
        ]

class PCAEmbedder(Embedder):

    def __init__(self, n_components=2):
        super().__init__(n_components)

    def run(self):
        pca = PCA(n_components=self.n_components)
        self.Y = pca.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

class TSNEEmbedder(Embedder):

    def __init__(self, n_components=2, perplexity=30, n_iter=1000):
        super().__init__(n_components)
        self.perplexity = perplexity

    def run(self):
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity)
        self.Y = tsne.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.extend([
            ('perplexity', float, 30.0),
            ('n_iter', int, 10)
        ])
        return params
