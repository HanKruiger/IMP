from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from model.dataset import Dataset, Sampling
from operators.operator import Operator
from sklearn.neighbors import NearestNeighbors

class RandomSampler(Operator):

    def __init__(self):
        super().__init__()

    def compute_support(self, representatives, points):
        nn = NearestNeighbors(1)
        nn.fit(representatives)
        labels = nn.kneighbors(points, return_distance=False).flatten()
        assert(len(labels) == points.shape[0])
        support = dict()
        for i in range(representatives.shape[0]):
            idcs = np.where(labels == i)[0]
            if len(idcs) == 0:
                print('No support for {}'.format(i))
            support[i] = idcs
        return support

    def run(self):
        in_dataset = self.input()['parent']

        X = in_dataset.data()

        N = X.shape[0]
        k = self.parameters()['k']
        idcs = np.random.choice(N, k, replace=False)
        Y = X[idcs, :]
        
        out_dataset = Sampling('S({})'.format(in_dataset.name()), parent=in_dataset, idcs=idcs, hidden=in_dataset.hidden_features())
        if self.parameters()['save_support']:
            Y_use, _ = Operator.hide_features(Y, in_dataset.hidden_features())
            X_use, _ = Operator.hide_features(X, in_dataset.hidden_features())
            out_dataset.set_support(self.compute_support(Y_use, X_use))
        self.set_output(out_dataset)

    @classmethod
    def parameters_description(cls):
        return {
            'k': (int, 1000),
            'save_support': (bool, False)
        }

    @classmethod
    def input_description(cls):
        return {
            'parent': Dataset
        }
