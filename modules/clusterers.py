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

        # Filter out the non-hidden features that are used as input for the clustering
        X_use = in_dataset.X[:, features]
        # Do the clustering
        Y, X_labels = self.cluster(X_use)

        n_clusters = Y.shape[0]

        support = self.find_support(X_labels, n_clusters)

        # Assign averaged values (over the support) to the representatives' hidden features
        X_hidden = in_dataset.X[:, hidden_features]
        Y_hidden = np.zeros((n_clusters, n_hidden_features))
        for i in range(n_clusters):
            Y_hidden[i, :] = X_hidden[support[i], :].mean(axis=0)
        
        # Concatenate the output of the clustering with the averaged hidden features
        Y = np.column_stack([Y, Y_hidden])

        out_dataset = Dataset(in_dataset.name + '_clus', parent=in_dataset, relation='clus', X=Y, support=support, hidden=hidden_features)
        self.set_output(out_dataset)

    @abc.abstractmethod
    def embed(self, X):
        """Method that should do the clustering"""

    @classmethod
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
        return {
            'dataset': (Dataset, True)
        }


class KMeansClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def cluster(self, X):
        kmeans = KMeans(**self.parameters())
        kmeans.fit(X)

        return kmeans.cluster_centers_, kmeans.labels_

    
    @classmethod
    def parameters_description(cls):
        return {
            'n_clusters': (int, 1000),
            'n_jobs': (int, 1)
        }

class MiniBatchKMeansClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def cluster(self, X):
        mbkmeans = MiniBatchKMeans(**self.parameters())
        mbkmeans.fit(X)

        return mbkmeans.cluster_centers_, mbkmeans.labels_

    
    @classmethod
    def parameters_description(cls):
        return {
            'n_clusters': (int, 1000),
            'batch_size': (int, 1000),
            'max_iter': (int, 100),
            'reassignment_ratio': (int, 0.01)
        }
