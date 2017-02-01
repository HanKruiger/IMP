from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import Dataset

import abc
import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


class Clusterer(QThread):

    def __init__(self):
        super().__init__()

    def set_input(self, dataset):
        self.in_dataset = dataset

    def set_parameters(self, parameters):
        self.parameters = parameters

    @staticmethod
    def find_support(labels, n_clusters):
        support = dict()
        for i in range(n_clusters):
            idcs = np.where(labels == i)
            support[i] = idcs
        return support

    @abc.abstractmethod
    def run(self):
        """Method that should run the clustering"""

    @classmethod
    @abc.abstractmethod
    def parameters_description(cls):
        """Method that should run parameters needed for the embedding,
        along with their types and default values. """


class KMeansClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def run(self):
        kmeans = KMeans(**self.parameters)
        kmeans.fit(self.in_dataset.X)
        Y = kmeans.cluster_centers_
        X_labels = kmeans.labels_

        support = self.find_support(X_labels, self.parameters['n_clusters'])

        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='kmeans', X=Y, support=support)

    
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
        kmeans = MiniBatchKMeans(**self.parameters)
        kmeans.fit(self.in_dataset.X)
        Y = kmeans.cluster_centers_
        X_labels = kmeans.labels_

        support = self.find_support(X_labels, self.parameters['n_clusters'])

        self.out_dataset = Dataset(self.in_dataset.name, parent=self.in_dataset, relation='mb_kmeans', X=Y, support=support)

    
    @classmethod
    def parameters_description(cls):
        return [
            ('n_clusters', int, 1000),
            ('batch_size', int, 1000)
        ]

class ClusterReplicator(Clusterer):
    def __init__(self, clustering):
        super().__init__()
        self.support = clustering.support

    def run(self):
        Y = np.zeros((len(self.support), self.in_dataset.X.shape[1]))
        for idx, support in self.support.items():
            print(idx)
            print(support)
            # TODO: Compute which row from in_dataset.X should be placed to represent the cluster.
        self.out_dataset = None
