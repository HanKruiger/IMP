from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model.dataset import Dataset, Clustering
from operators.operator import Operator

import abc
import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

class Clusterer(Operator):

    def __init__(self):
        super().__init__()

    @staticmethod
    def find_support(labels, n_clusters):
        support = dict()
        for i in range(n_clusters):
            idcs = np.where(labels == i)[0]
            if len(idcs) == 0:
                pass
                # print('No support for {}'.format(idcs))
            support[i] = idcs
        return support

    def run(self):
        in_dataset = self.input()['parent']
        n_hidden_features = self.parameters()['n_hidden_features']

        X_use, X_hidden = Operator.hide_features(in_dataset.data(), n_hidden_features)

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

        out_dataset = Clustering('C({})'.format(in_dataset.name()), in_dataset, Y, support, hidden=n_hidden_features)
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
            'n_jobs': (int, 1),
            'init': (str, 'k-means++')
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

class DBSCANClusterer(Clusterer):

    def __init__(self):
        super().__init__()

    def cluster(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        
        dbscan = DBSCAN(**parameters)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Take average of items in the cluster
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            centroids[i, :] = X[labels == i, :].mean(axis=0)

        return centroids, labels

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'eps': (float, 0.5),
            'min_samples': (int, 5),
            'metric': (str, 'euclidean'),
            'algorithm': (str, 'auto'),
            'leaf_size': (int, 30)
        })
        return desc

class BirchClusterer(Clusterer):

    def __init__(self):
        super().__init__()


    def cluster(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        
        birch = Birch(**parameters)
        labels = birch.fit_predict(X)
        
        # Take average of items in the cluster
        centroids = np.zeros((self.parameters()['n_clusters'], X.shape[1]))
        for i in range(self.parameters()['n_clusters']):
            centroids[i, :] = X[labels == i, :].mean(axis=0)

        return centroids, labels

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_clusters': (int, 1000)
        })
        return desc

class KNNAgglomerativeClusterer(Clusterer):

    def __init__(self):
        super().__init__()


    def cluster(self, X):
        parameters = self.parameters().copy()
        del parameters['n_hidden_features']
        
        knn_graph = kneighbors_graph(X, parameters['k'], include_self=False)

        del parameters['k']
        parameters['connectivity'] = knn_graph

        aggclu = AgglomerativeClustering(**parameters)
        labels = aggclu.fit_predict(X)
        
        # Take average of items in the cluster
        centroids = np.zeros((self.parameters()['n_clusters'], X.shape[1]))
        for i in range(self.parameters()['n_clusters']):
            centroids[i, :] = X[labels == i, :].mean(axis=0)

        return centroids, labels

    @classmethod
    def parameters_description(cls):
        desc = super().parameters_description()
        desc.update({
            'n_clusters': (int, 1000),
            'linkage': (str, 'ward'),
            'compute_full_tree': (bool, True),
            'k': (int, 20)
        })
        return desc