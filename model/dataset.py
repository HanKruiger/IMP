from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import time
import os
import abc
import numpy as np
import numpy.ma as ma
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
import time
import annoy
from model.hypersphere import HyperSphere


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

    def set_name(self, new_name):
        self._name = new_name

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

    def __add__(self, other, sort=True):
        assert(isinstance(other, Dataset))
        assert(other.n_dimensions() == self.n_dimensions())
        assert(np.intersect1d(other.indices(), self.indices()).size == 0)

        root_idcs = np.concatenate((self.indices(), other.indices()), axis=0)
        data = np.concatenate((self.data(), other.data()), axis=0)
        
        if sort:
            order = np.argsort(root_idcs)
            root_idcs = root_idcs[order]
            data = data[order, :]
        
        union = Dataset(data, root_idcs, name='Union')
        return union

    def __sub__(self, other):
        idcs_in_root = np.setdiff1d(self.indices(), other.indices())
        # Assumes self.indices() is sorted.
        idcs_in_self = np.searchsorted(self.indices(), idcs_in_root)

        difference = self.select_indices(idcs_in_self)
        difference.set_name('Difference')
        return difference

    def select_logical(self, bool_values):
        assert(bool_values.size == self.n_points())

        # Make sure they're actually booleans, so slicing happens correctly.
        bool_values = np.array(bool_values, dtype=np.bool)

        root_idcs = self.indices()[bool_values]
        data = self.data()[bool_values, :]
        selection = Dataset(data, root_idcs, name='Logical selection')
        return selection

    def select_indices(self, indices):
        # Make sure they're actually integer indices, so slicing happens correctly.
        indices_in_self = np.array(indices, dtype=np.int)

        root_idcs = self.indices()[indices_in_self]
        data = self.data()[indices_in_self, :]
        selection = Dataset(data, root_idcs, name='Logical selection')
        return selection

    def random_sampling(self, M, sort=True):
        if M < self.n_points():
            idcs_in_self = np.random.choice(self.n_points(), M, replace=False)
            if sort:
                idcs_in_self.sort()
        else:
            from warnings import warn
            warn('Sample size larger than (or eq. to) source. Using all source samples.', RuntimeWarning)
            idcs_in_self = np.arange(self.n_points())

        root_idcs = self.indices()[idcs_in_self]
        data = self.data()[idcs_in_self, :]
        sampling = Dataset(data, root_idcs, name='Random sampling')
        return sampling

    def radius(self, smooth=False):
        if smooth:
            return self.data().std(axis=0).max()
        else:
            return np.abs(self.data()).max()

    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            self._centroid = self.data().mean(axis=0)
            return self.centroid()

    def bounding_hypersphere(self, smooth=False):
        return HyperSphere(self.centroid(), self.radius(smooth=smooth))

    def knn_pointset(self, n_samples, query_idcs=None, query_dataset=None, remove_query_points=False, k=2, method='tree', sort=True, verbose=2):
        # We're using this function recursively, so timing is more complex.
        if not hasattr(self, 'knn_pointset_t_0'):
            self.knn_pointset_t_0 = time.time()

        if method == 'tree':
            if query_idcs is not None:
                indices = np.zeros((query_idcs.size, k), dtype=np.int)
                dists = np.zeros_like(indices)
                t_0 = time.time()
                for i, idx in enumerate(query_idcs):
                    res = self.tree().get_nns_by_item(idx, k, include_distances=True)
                    indices[i, :] = res[0]
                    dists[i, :] = res[1]
                if verbose >= 2:
                    print('\tQuerying tree took {:.1f} ms.'.format(1000 * (time.time() - t_0)))
                t_0 = time.time()
                mask = np.zeros_like(indices, dtype=np.bool)
                if remove_query_points:
                    for i in range(indices.shape[0]):
                        mask[i, :] = np.in1d(indices[i, :], query_idcs)
                if verbose >= 2:
                    print('\tMasking indices took {:.1f} ms.'.format(1000 * (time.time() - t_0)))
            elif query_dataset is not None:
                indices = np.zeros((query_dataset.n_points(), k), dtype=np.int)
                dists = np.zeros_like(indices)
                t_0 = time.time()
                for i in range(query_dataset.n_points()):
                    res = self.tree().get_nns_by_vector(query_dataset.data()[i, :], k, include_distances=True)
                    indices[i, :] = res[0]
                    dists[i, :] = res[1]
                if verbose >= 2:
                    print('\tQuerying tree took {:.1f} ms.'.format(1000 * (time.time() - t_0)))
                t_0 = time.time()
                mask = np.zeros_like(indices, dtype=np.bool)
                if remove_query_points:
                    for i in range(indices.shape[0]):
                        mask[i, :] = np.in1d(self.indices()[indices[i, :]], query_dataset.indices())
                if verbose >= 2:
                    print('\tMasking indices took {:.1f} ms.'.format(1000 * (time.time() - t_0)))
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
                k_new = min(2 * k, self.n_points())
                print('\t{}-nn was too few (only {} unique results, needed {}) retrying with {}-nn...'.format(k, unique_idcs.size, n_samples, k_new))
                return self.knn_pointset(n_samples=n_samples, query_idcs=query_idcs, query_dataset=query_dataset, remove_query_points=remove_query_points, k=k_new, method=method, verbose=verbose)

            # Reduce to the smallest distance per unique index.
            t_0 = time.time()
            min_dists = np.zeros_like(unique_idcs)
            for i, (idx_start, count) in enumerate(zip(idx_starts, counts)):
                min_dists[i] = dists_c[idx_start:idx_start + count].min()
            if verbose >= 2:
                print('\tMerging indices and distances took {:.1f} ms.'.format(1000 * (time.time() - t_0)))

            if unique_idcs.size > n_samples:
                # Use only the n_samples closest samples for the result.
                closest_idcs = np.argpartition(min_dists, n_samples)[:n_samples]
                idcs_in_self = unique_idcs[closest_idcs]
            else:
                idcs_in_self = unique_idcs
        elif method == 'bruteforce':
            if query_idcs is not None:
                query = self.data()[query_idcs, :]
            elif query_dataset is not None:
                query = query_dataset.data()
            # Compute smallest distances from all root points to all query points.
            dists = cdist(query, self.data(), metric='euclidean').min(axis=0)
            # Retrieve indices (in source!) where the distances are smallest
            if n_samples == dists.size:
                idcs_in_self = np.arange(n_samples)
            else:
                idcs_in_self = np.argpartition(dists, n_samples)[:n_samples]
        else:
            raise ValueError('Argument \'method\' must be either \'tree\' or \'bruteforce\'.')

        if sort:
            idcs_in_self.sort()

        data = self.data()[idcs_in_self, :]
        idcs_in_root = self.indices()[idcs_in_self]
        dataset = Dataset(data, idcs_in_root, name='KNN pointset ({})'.format(method))

        if verbose:
            print('knn_pointset ({}) took {:.1f} ms.'.format(method, 1000 * (time.time() - self.knn_pointset_t_0)))
        del self.knn_pointset_t_0

        return dataset

