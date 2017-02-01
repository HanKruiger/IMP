from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import Dataset

import abc
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding


class Embedder(QThread):

    def __init__(self):
        super().__init__()

    def set_input(self, dataset):
        self.in_dataset = dataset

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
    @abc.abstractmethod
    def parameters_description(cls):
        """Method that should run parameters needed for the embedding,
        along with their types and default values. """


class PCAEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        pca = PCA(**self.parameters)
        Y = pca.fit_transform(self.in_dataset.X)
        Y = self.normalize(Y)
        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='pca', X=Y)

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2)
        ]


class TSNEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def run(self):
        tsne = TSNE(**self.parameters)
        Y = tsne.fit_transform(self.in_dataset.X)
        Y = self.normalize(Y)
        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='tsne', X=Y)

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
        lle = LocallyLinearEmbedding(**self.parameters)
        Y = lle.fit_transform(self.in_dataset.X)
        Y = self.normalize(Y)
        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='lle', X=Y)

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
        lle = SpectralEmbedding(**self.parameters)
        Y = lle.fit_transform(self.in_dataset.X)
        Y = self.normalize(Y)
        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='spectral', X=Y)

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
        lle = MDS(**self.parameters)
        Y = lle.fit_transform(self.in_dataset.X)
        Y = self.normalize(Y)
        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='mds', X=Y)

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
        lle = Isomap(**self.parameters)
        Y = lle.fit_transform(self.in_dataset.X)
        Y = self.normalize(Y)
        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='isomap', X=Y)

    @classmethod
    def parameters_description(cls):
        return [
            ('n_components', int, 2),
            ('n_neighbors', int, 5)
        ]
