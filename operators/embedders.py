from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model.dataset import Dataset, Embedding
from operators.operator import Operator

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

    def run(self):
        in_dataset = self.input()['parent']
        n_hidden_features = self.parameters()['n_hidden_features']

        X_use, X_hidden = Operator.hide_features(in_dataset.data(), n_hidden_features)

        # Do the embedding
        Y = self.embed(X_use)

        # Concatenate the output of the embedding with the hidden features
        Y = np.column_stack([Y, X_hidden])

        out_dataset = Embedding('E({})'.format(in_dataset.name()), in_dataset, Y, hidden=n_hidden_features)
        self.set_output(out_dataset)

    @abc.abstractmethod
    def embed(self, X):
        """Method that should do the embedding"""

    @classmethod
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
        return {
            'parent': Dataset
        }

    @classmethod
    def parameters_description(cls):
        return {
            'n_hidden_features': (int, None)
        }


class PCAEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        pca = PCA(**parameters)
        Y = pca.fit_transform(X)
        return Y

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_components': (int, 2)
        })
        return desc


class TSNEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        tsne = TSNE(**parameters)
        Y = tsne.fit_transform(X)
        return Y

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_components': (int, 2),
            'perplexity': (float, 30.0),
            'n_iter': (int, 200)
        })
        return desc


class LLEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        lle = LocallyLinearEmbedding(**parameters)
        Y = lle.fit_transform(X)
        return Y

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_components': (int, 2),
            'n_neighbors': (int, 5)
        })
        return desc


class SpectralEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        se = SpectralEmbedding(**parameters)
        Y = se.fit_transform(X)
        return Y

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_components': (int, 2),
            'n_neighbors': (int, 5)
        })
        return desc


class MDSEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        mds = MDS(**parameters)
        Y = mds.fit_transform(X)
        return Y

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_components': (int, 2),
            'max_iter': (int, 300),
            'metric': (int, 1)
        })
        return desc


class IsomapEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        isomap = Isomap(**parameters)
        Y = isomap.fit_transform(X)
        return Y

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_components': (int, 2),
            'n_neighbors': (int, 5)
        })
        return desc


class LAMPEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        representatives_2d = self.input()['representatives_2d']
        representatives_nd = self.input()['representatives_nd']

        Y_s, _ = Operator.hide_features(representatives_2d.data(), self.parameters()['n_hidden_features'])
        X_s, _ = Operator.hide_features(representatives_nd.data(), self.parameters()['n_hidden_features'])

        N = X.shape[0]
        n = Y_s.shape[1]

        Y = np.zeros((N, n))

        for i in np.arange(N):
            x = X[i, :]

            alphas = np.sum((X_s - x)**2, axis=1)

            x_tilde = (alphas * X_s.T).sum(axis=1) / alphas.sum()
            y_tilde = (alphas * Y_s.T).sum(axis=1) / alphas.sum()

            A_T = alphas * (X_s - x_tilde).T
            B = (alphas * (Y_s - y_tilde).T).T

            U, _, V = np.linalg.svd(A_T.dot(B), full_matrices=False)

            Y[i, :] = (x - x_tilde).dot(U.dot(V)) + y_tilde

        return Y

    @classmethod
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
        return {
            'parent': Dataset,
            'representatives_2d': Dataset,
            'representatives_nd': Dataset
        }
