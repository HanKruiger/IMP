from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import time
import os
import abc
import numpy as np
import numpy.ma as ma
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
import time
import annoy


class Dataset(QObject):

    def __init__(self, data, idcs, name='X', is_root=False):
        super().__init__()
        self._data = data
        self._name = name  # Watch it: recursive names are memory hogs
        self._idcs = idcs
        self._is_root = is_root

        # Not the prettiest way, I think. Needs thought.
        if is_root:
            Dataset.root = self

    @staticmethod
    def set_root_labels(labels):
        try:
            if labels.shape[0] != Dataset.root.n_points():
                print('Observations in labels don\'t match with root dataset. Labels not set.')
                return False
        except AttributeError:
            print('Root dataset does not exist yet. Labels not set.')
            return False

        Dataset.labels = labels
        return True

    def tree(self):
        try:
            return self._tree

        except AttributeError:
            # Build the tree
            print('Building ANN trees.')
            t_0 = time.time()
            tree = annoy.AnnoyIndex(self.n_dimensions())
            for i in range(self.n_points()):
                tree.add_item(i, self.data()[i, :])
            tree.build(10)
            self._tree = tree
            print('Building trees took {:.2f} s.'.format(time.time() - t_0))

            return self.tree()

    def name(self):
        return self._name

    def n_points(self):
        return self.data().shape[0]

    def n_dimensions(self):
        if len(self.data().shape) < 2:
            return 1
        else:
            return self.data().shape[1]

    def destroy(self):
        if self._data is not None:
            del self._data

    def data(self):
        return self._data

    def indices(self):
        return self._idcs

    def root_indices_to_own(self, root_idcs):
        try:
            return np.array([self._root_idcs_lookup[i] for i in root_idcs])
        except AttributeError:
            self._root_idcs_lookup = defaultdict(lambda: -1)
            for own_idx, root_idx in enumerate(self.indices()):
                self._root_idcs_lookup[root_idx] = own_idx
            return self.root_indices_to_own(root_idcs)

    def radius(self):
        return self.data().std(axis=0).max()

    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            self._centroid = self.data().mean(axis=0)
            return self.centroid()

    def knn_pointset(self, n_samples, query_idcs=None, query_dataset=None, remove_query_points=False, k=2, method='tree', sort=True, verbose=2):
        if query_idcs is not None:
            indices = np.zeros((query_idcs.size, k), dtype=np.int)
            dists = np.zeros_like(indices)
            for i, idx in enumerate(query_idcs):
                res = self.tree().get_nns_by_item(idx, k, include_distances=True)
                indices[i, :] = res[0]
                dists[i, :] = res[1]
            mask = np.zeros_like(indices, dtype=np.bool)
            if remove_query_points:
                for i in range(indices.shape[0]):
                    mask[i, :] = np.in1d(indices[i, :], query_idcs)
        elif query_dataset is not None:
            indices = np.zeros((query_dataset.n_points(), k), dtype=np.int)
            dists = np.zeros_like(indices)
            for i in range(query_dataset.n_points()):
                res = self.tree().get_nns_by_vector(query_dataset.data()[i, :], k, include_distances=True)
                indices[i, :] = res[0]
                dists[i, :] = res[1]
            mask = np.zeros_like(indices, dtype=np.bool)
            if remove_query_points:
                for i in range(indices.shape[0]):
                    mask[i, :] = np.in1d(self.indices()[indices[i, :]], query_dataset.indices())
        else:
            raise ValueError('Either query_idcs or query_dataset must be given.')

        m_indices = ma.masked_array(indices, mask=mask)
        m_dists = ma.masked_array(dists, mask=mask)
        indices_c = m_indices.compressed()
        dists_c = m_dists.compressed()

        idx_sort = np.argsort(indices_c)
        indices_c = indices_c[idx_sort]
        dists_c = dists_c[idx_sort]

        unique_idcs, idx_starts, counts = np.unique(indices_c, return_index=True, return_counts=True)

        if unique_idcs.size < n_samples:
            print('\t{}-nn was too few (only {} unique results, needed {}) retrying with {}-nn...'.format(k, unique_idcs.size, n_samples, 2*k))
            return self.knn_pointset(n_samples=n_samples, query_idcs=query_idcs, query_dataset=query_dataset, remove_query_points=remove_query_points, k=2*k, method=method, verbose=verbose)

        # Reduce to the smallest distance per unique index.
        min_dists = np.zeros_like(unique_idcs)
        for i, (idx_start, count) in enumerate(zip(idx_starts, counts)):
            min_dists[i] = dists_c[idx_start:idx_start + count].min()

        if unique_idcs.size > n_samples:
            # Use only the n_samples closest samples for the result.
            closest_idcs = np.argpartition(min_dists, n_samples)[:n_samples]
            idcs_in_self = unique_idcs[closest_idcs]
        else:
            idcs_in_self = unique_idcs

        if sort:
            idcs_in_self.sort()

        data = self.data()[idcs_in_self, :]
        idcs_in_root = self.indices()[idcs_in_self]
        dataset = Dataset(data, idcs_in_root, name='KNN pointset')
        return dataset
