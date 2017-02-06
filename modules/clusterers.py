from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import Dataset
from modules.operator import Operator

import abc
import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


class Clusterer(Operator):

    def __init__(self):
        super().__init__()

    @staticmethod
    def find_support(labels, n_clusters):
        support = dict()
        for i in range(n_clusters):
            idcs = np.where(labels == i)[0]
            if len(idcs) == 0:
                print('No support for {}'.format(idcs))
            support[i] = idcs
        return support

    @classmethod
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
        return [
            ('dataset', Dataset)
        ]


class KMeansClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def run(self):
        kmeans = KMeans(**self.parameters())
        kmeans.fit(self.input()[0].X)
        Y = kmeans.cluster_centers_
        X_labels = kmeans.labels_

        support = self.find_support(X_labels, self.parameters()['n_clusters'])

        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='kmeans', X=Y, support=support))

    
    @classmethod
    def parameters_description(cls):
        return [
            ('n_clusters', int, 1000),
            ('n_jobs', int, 1)
        ]

class MiniBatchKMeansClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def run(self):
        kmeans = MiniBatchKMeans(**self.parameters())
        kmeans.fit(self.input()[0].X)
        Y = kmeans.cluster_centers_
        X_labels = kmeans.labels_

        support = self.find_support(X_labels, self.parameters()['n_clusters'])

        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='mb_kmeans', X=Y, support=support))

    
    @classmethod
    def parameters_description(cls):
        return [
            ('n_clusters', int, 1000),
            ('batch_size', int, 1000),
            ('max_iter', int, 100),
            ('reassignment_ratio', int, 0.01)
        ]
