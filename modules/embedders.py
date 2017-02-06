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

    def run(self):
        in_dataset = self.input()[0][0]
        hidden_features = self.input()[0][1]

        n_hidden_features = len(hidden_features)
        n_features = in_dataset.m - n_hidden_features

        mask = np.ones(in_dataset.m, dtype=bool)
        mask[hidden_features,] = False
        features = np.arange(in_dataset.m)[mask,]
        del mask

        assert(len(features) == n_features)

        # Filter out the subset of features that are used as input for the embedding
        X_use = in_dataset.X[:, features]
        # Do the embedding
        Y = self.embed(X_use)

        embed_in_dim = X_use.shape[1]
        embed_out_dim = Y.shape[1]
        reduction = embed_in_dim - embed_out_dim

        # Filter out the subset that isn't used
        X_hidden = in_dataset.X[:, hidden_features]
        
        # Concatenate the output of the embedding with the not-considered features
        Y = np.column_stack([Y, X_hidden])

        out_dataset = Dataset(in_dataset.name + '_emb', parent=in_dataset, relation='emb', X=Y, hidden=np.arange(embed_out_dim, embed_out_dim + n_hidden_features))
        self.set_output(out_dataset)

    @abc.abstractmethod
    def embed(self, X):
        """Method that should do the embedding"""

    @classmethod
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
        return {
            'dataset': (Dataset, True)
        }

class PCAEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        pca = PCA(**self.parameters())
        Y = pca.fit_transform(X)
        Y = self.normalize(Y)
        return Y

    @classmethod
    def parameters_description(cls):
        return {
            'n_components': (int, 2)
        }


class TSNEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        tsne = TSNE(**self.parameters())
        Y = tsne.fit_transform(X)
        Y = self.normalize(Y)
        return Y

    @classmethod
    def parameters_description(cls):
        return {
            'n_components': (int, 2),
            'perplexity': (float, 30.0),
            'n_iter': (int, 200)
        }


class LLEEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        lle = LocallyLinearEmbedding(**self.parameters())
        Y = lle.fit_transform(X)
        Y = self.normalize(Y)
        return Y

    @classmethod
    def parameters_description(cls):
        return {
            'n_components': (int, 2),
            'n_neighbors': (int, 5)
        }


class SpectralEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        se = SpectralEmbedding(**self.parameters())
        Y = se.fit_transform(X)
        Y = self.normalize(Y)
        return Y

    @classmethod
    def parameters_description(cls):
        return {
            'n_components': (int, 2),
            'n_neighbors': (int, 5)
        }


class MDSEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        mds = MDS(**self.parameters())
        Y = mds.fit_transform(X)
        Y = self.normalize(Y)
        return Y

    @classmethod
    def parameters_description(cls):
        return {
            'n_components': (int, 2),
            'max_iter': (int, 300),
            'metric': (int, 1)
        }


class IsomapEmbedder(Embedder):

    def __init__(self):
        super().__init__()

    def embed(self, X):
        isomap = Isomap(**self.parameters())
        Y = isomap.fit_transform(X)
        Y = self.normalize(Y)
        return Y

    @classmethod
    def parameters_description(cls):
        return {
            'n_components': (int, 2),
            'n_neighbors': (int, 5)
        }
