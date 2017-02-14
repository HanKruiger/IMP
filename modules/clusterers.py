from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import Dataset, Clustering
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

    def run(self):
        in_dataset = self.input()['parent']
        n_hidden_features = self.parameters()['n_hidden_features']

        X_use, X_hidden = Operator.hide_features(in_dataset.X, n_hidden_features)

        # Do the clustering
        Y, X_labels = self.cluster(X_use)

        n_clusters = Y.shape[0]

        support = self.find_support(X_labels, n_clusters)

        # Assign averaged values (over the support) to the representatives' hidden features
        Y_hidden = np.zeros((n_clusters, n_hidden_features))
        for i in range(n_clusters):
            Y_hidden[i, :] = X_hidden[support[i], :].mean(axis=0)

        # Concatenate the output of the clustering with the averaged hidden features
        Y = np.column_stack([Y, Y_hidden])

        out_dataset = Clustering(in_dataset.name() + 'c', in_dataset, Y, support, hidden=n_hidden_features)
        self.set_output(out_dataset)

    @abc.abstractmethod
    def cluster(self, X):
        """Method that should do the clustering"""

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


class KMeansClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def cluster(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        kmeans = KMeans(**parameters)
        kmeans.fit(X)

        return kmeans.cluster_centers_, kmeans.labels_

    
    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_clusters': (int, 1000),
            'n_jobs': (int, 1)
        })
        return desc

class MiniBatchKMeansClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def cluster(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        mbkmeans = MiniBatchKMeans(**parameters)
        mbkmeans.fit(X)

        return mbkmeans.cluster_centers_, mbkmeans.labels_

    
    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_clusters': (int, 1000),
            'batch_size': (int, 1000),
            'max_iter': (int, 100),
            'reassignment_ratio': (int, 0.01)
        })
        return desc
