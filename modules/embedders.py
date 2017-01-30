from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding

class Embedder(QThread):

    def __init__(self):
        super().__init__()

    def set_input(self, X):
        self.X = X

    def set_parameters(self, parameters):
        self.parameters = parameters

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
    def parameters_description(cls):
        return [
            ('n_components', int, 2)
        ]

class PCAEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        pca = PCA(**self.parameters)
        self.Y = pca.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

class TSNEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        tsne = TSNE(**self.parameters)
        self.Y = tsne.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

    @classmethod
    def parameters_description(cls):
        params = super().parameters_description()
        params.extend([
            ('perplexity', float, 30.0),
            ('n_iter', int, 200)
        ])
        return params


class LLEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = LocallyLinearEmbedding(**self.parameters)
        self.Y = lle.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

    @classmethod
    def parameters_description(cls):
        params = super().parameters_description()
        params.extend([
            ('n_neighbors', int, 5),
        ])
        return params

class SpectralEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = SpectralEmbedding(**self.parameters)
        self.Y = lle.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

    @classmethod
    def parameters_description(cls):
        params = super().parameters_description()
        params.extend([
            ('n_neighbors', int, 5)
        ])
        return params

class MDSEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = MDS(**self.parameters)
        self.Y = lle.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

    @classmethod
    def parameters_description(cls):
        params = super().parameters_description()
        params.extend([
            ('max_iter', int, 300),
            ('metric', int, 1)
        ])
        return params

class IsomapEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = Isomap(**self.parameters)
        self.Y = lle.fit_transform(self.X)
        self.Y = self.normalize(self.Y)

    @classmethod
    def parameters_description(cls):
        params = super().parameters_description()
        params.extend([
            ('n_neighbors', int, 5)
        ])
        return params
