from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import time
import os
import abc
import numpy as np
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
        self._name = name # Watch it: recursive names are memory hogs
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

    @staticmethod
    def root_tree():
        try:
            return Dataset.annoy_tree
        except AttributeError:
            # Build the tree
            t_0 = time.time()
            tree = annoy.AnnoyIndex(Dataset.root.n_dimensions())
            for i in range(Dataset.root.n_points()):
                tree.add_item(i, Dataset.root.data()[i, :])
            tree.build(10)
            Dataset.annoy_tree = tree

            print('Computing tree took {:.2f} s.'.format(time.time() - t_0))
            
            # Try again.
            return Dataset.root_tree()
            

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
