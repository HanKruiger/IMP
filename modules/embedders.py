from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import Dataset
from modules.operator import Operator

import abc
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding


class Embedder(Operator):

    def __init__(self):
        super().__init__()

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
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
        return [
            ('dataset', Dataset)
        ]

class PCAEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        pca = PCA(**self.parameters())
        Y = pca.fit_transform(self.input()[0].X)
        Y = self.normalize(Y)
        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='pca', X=Y))

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2)
        ]


class TSNEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        tsne = TSNE(**self.parameters())
        Y = tsne.fit_transform(self.input()[0].X)
        Y = self.normalize(Y)
        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='tsne', X=Y))

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2),
            ('perplexity', float, 30.0),
            ('n_iter', int, 200)
        ]


class LLEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = LocallyLinearEmbedding(**self.parameters())
        Y = lle.fit_transform(self.input()[0].X)
        Y = self.normalize(Y)
        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='lle', X=Y))

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2),
            ('n_neighbors', int, 5)
        ]


class SpectralEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = SpectralEmbedding(**self.parameters())
        Y = lle.fit_transform(self.input()[0].X)
        Y = self.normalize(Y)
        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='spectral', X=Y))

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2),
            ('n_neighbors', int, 5)
        ]


class MDSEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = MDS(**self.parameters())
        Y = lle.fit_transform(self.input()[0].X)
        Y = self.normalize(Y)
        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='mds', X=Y))

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2),
            ('max_iter', int, 300),
            ('metric', int, 1)
        ]


class IsomapEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        lle = Isomap(**self.parameters())
        Y = lle.fit_transform(self.input()[0].X)
        Y = self.normalize(Y)
        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='isomap', X=Y))

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2),
            ('n_neighbors', int, 5)
        ]
