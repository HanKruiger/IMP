from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from model.dataset import Dataset, Sampling
from operators.operator import Operator
from sklearn.neighbors import NearestNeighbors


class Sampler(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        in_dataset = self.input()['parent']
        n_hidden_features = self.parameters()['n_hidden_features']

        X_use, _ = Operator.hide_features(in_dataset.data(), n_hidden_features)

        idcs = self.sample(X_use)

        out_dataset = Sampling('S({})'.format(in_dataset.name()), parent=in_dataset, idcs=idcs, hidden=n_hidden_features)
        if self.parameters()['save_support']:
            Y_use, _ = Operator.hide_features(out_dataset.data(), n_hidden_features)
            out_dataset.set_support(Sampler.compute_support(Y_use, X_use))
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
            'parent': Dataset
        }


class RandomSampler(Sampler):

    def __init__(self):
        super().__init__()

    def sample(self, X):
        N = X.shape[0]
        k = self.parameters()['k']
        idcs = np.random.choice(N, k, replace=False)
        return idcs

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'k': (int, 1000)
        })
        return desc
