from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from model.dataset import Dataset, Sampling
from operators.operator import Operator
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from numpy.linalg import svd

class Sampler(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        in_dataset = self.input()['parent']
        try:
            n_hidden_features = self.parameters()['n_hidden_features']
        except KeyError:
            n_hidden_features = in_dataset.hidden_features()

        X_use, _ = Operator.hide_features(in_dataset.data(), n_hidden_features)

        idcs = self.sample(X_use)

        out_dataset = Sampling('S({})'.format(in_dataset.name()), parent=in_dataset, idcs=idcs, hidden=n_hidden_features)
        
        try:
            if self.parameters()['save_support']:
                Y_use, _ = Operator.hide_features(out_dataset.data(), n_hidden_features)
                out_dataset.set_support(Sampler.compute_support(Y_use, X_use))
        except KeyError:
            pass

        try:
            if self.input()['sibling'] is not None:
                in_sibling = self.input()['sibling']
                assert(in_sibling.N == in_dataset.N)
                out_sibling = Sampling('S({})'.format(in_sibling.name()), parent=in_sibling, idcs=idcs, hidden=in_sibling.hidden_features())
                self.set_outputs([out_dataset, out_sibling])
            else:
                self.set_output(out_dataset)
        except KeyError:
            self.set_output(out_dataset)

    @abc.abstractmethod
    def sample(self, X):
        """Method that should do the sampling"""

    @staticmethod
    def compute_support(representatives, points):
        nn = NearestNeighbors(1)
        nn.fit(representatives)
        labels = nn.kneighbors(points, return_distance=False).flatten()

        support = dict()
        for i in range(representatives.shape[0]):
            idcs = np.where(labels == i)[0]
            if len(idcs) == 0:
                print('No support for {}'.format(i))
            support[i] = idcs
        return support

    @classmethod
    def parameters_description(cls):
        return {
            'n_hidden_features': (int, 0),
            'save_support': (bool, False)
        }

    @classmethod
    def input_description(cls):
        return {
            'parent': Dataset,
            'sibling': Dataset # Same subsampling will be applied to sibling.
        }


class RandomSampler(Sampler):

    def __init__(self):
        super().__init__()

    def sample(self, X):
        N = X.shape[0]
        k = self.parameters()['n_samples']
        idcs = np.random.choice(N, k, replace=False)
        return idcs

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_samples': (int, 1000)
        })
        return desc

class SVDBasedSampler(Sampler):

    def __init__(self):
        super().__init__()

    def sample(self, X):
        try:
            k = self.parameters()['k']
        except KeyError:
            k = min(X.shape) // 2 + 1 # As suggested by Joia et al.

        c = self.parameters()['n_samples']

        if k < X.shape[1]:
            _, _, V_T = svds(X.T, k=k)
        else:
            _, _, V_T = svd(X.T, full_matrices=False)

        pi = (V_T ** 2).sum(axis=0)
        # Get the c indices with the c largest value in pi (in no particular order)
        idcs = np.argpartition(pi, -c)[-c:]

        return idcs

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_samples': (int, 1000),
            'k': (int, '')
        })
        return desc
