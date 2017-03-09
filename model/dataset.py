from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np


class Dataset(QObject):

    # Emitted when (child) embedding is finished
    has_new_child = pyqtSignal(object)

    def __init__(self, name, parent, X, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = parent.hidden_features()

        self._name = name
        self._parent = parent
        self._n_hidden_features = hidden

        self._workers = set()
        self._children = []
        self._q_item = None

        if X is not None:
            self._X = X

    def n_points(self):
        return self.data().shape[0]

    def n_dimensions(self):
        if len(self.data().shape) < 2:
            return 1
        else:
            return self.data().shape[1]

    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def destroy(self):
        del self._X
        if self._parent is not None:
            self._parent.remove_child(self)

    def parent(self):
        return self._parent

    def q_item(self):
        return self._q_item

    def set_q_item(self, q_item):
        self._q_item = q_item

    def child(self, idx):
        return self._children[idx]

    def child_count(self):
        return len(self._children)

    def add_child(self, child):
        self._children.append(child)
        self.has_new_child.emit(child)

    def remove_child(self, child):
        self._children.remove(child)

    def is_clustering(self):
        return self._is_clustering

    def data(self):
        return self._X

    def indices(self):
        if self.parent() is not None:
            return self.parent().indices()
        else:
            return np.arange(self.n_points())

    def hidden_features(self):
        return self._n_hidden_features

    def perform_operation(self, operator):
        # Somehow I need this local function...
        def handle_results(op):
            self.handle_operator_results(op)
        operator.has_results.connect(handle_results)

        operator.start()
        self._workers.add(operator)

    @pyqtSlot(object)
    def handle_operator_results(self, operator):
        # Fetch data from worker, and delete it
        result = operator.output()
        self._workers.remove(operator)  # Your services are no longer needed.

        if type(result) == tuple:
            for child in result:
                child.parent().add_child(child)
        else:
            result.parent().add_child(result)

    def normalize(self):
        Y = self.data()[:, :-self.hidden_features()].copy()

        # Translate s.t. smallest values for both x and y are 0.
        for dim in range(Y.shape[1]):
            Y[:, dim] += -Y[:, dim].min()

        # Scale s.t. max(max(x, y)) = 1 (while keeping the same aspect ratio!)
        scaling = 1 / (np.absolute(Y).max())
        Y *= scaling

        # Centralize the median
        Y -= np.median(Y, axis=0)
        self.data()[:, :-self.hidden_features()] = Y


class DatasetItem(QStandardItem):

    def __init__(self, text):
        super().__init__(text)

    def type(self):
        return QStandardItem.UserType

    def setData(self, data, role):
        if role == Qt.UserRole:
            self._data = data
            data.set_q_item(self)
            self.emitDataChanged()
        else:
            super().setData(data, role)

    def data(self, role):
        if role == Qt.UserRole:
            return self._data
        else:
            return super().data(role)


class InputDataset(Dataset):

    def __init__(self, name, X, hidden=None):
        super().__init__(name, None, X, hidden=hidden)

    def root(self):
        return self

    def root_data(self):
        return self.data()


class Clustering(Dataset):

    def __init__(self, name, parent, X, support, hidden=None):
        super().__init__(name, parent, X, hidden=hidden)
        self._support = support

    def support(self):
        return self._support


class Embedding(Dataset):

    def __init__(self, name, parent, X, hidden=None):
        super().__init__(name, parent, X, hidden=hidden)

    def root(self):
        return self.parent().root()

    def root_data(self):
        return self.parent().root_data()


class Selection(Dataset):

    def __init__(self, name, parent, idcs, hidden=None):
        self._idcs = idcs
        super().__init__(name, parent, None, hidden=hidden)

    def data(self):
        return self.parent().data()[self._idcs, :]

    def root(self):
        return self.parent().root()

    def root_data(self):
        return self.parent().root_data()[self._idcs, :]

    def indices(self):
        return self.parent().indices()[self._idcs]

    def destroy(self):
        del self._idcs
        if self._parent is not None:
            self._parent.remove_child(self)


class Sampling(Selection):

    def __init__(self, name, parent, idcs, hidden=None):
        super().__init__(name, parent, idcs, hidden=hidden)

    def set_support(self, support):
        self._support = support

    def support(self):
        return self._support


class Merging(Dataset):

    def __init__(self, name, parent, X, hidden=None):
        super().__init__(name, parent, X, hidden=hidden)

    def root(self):
        return self

    def root_data(self):
        return self.data()


class Union(Dataset):

    def __init__(self, parent_1, parent_2, name=None, hidden=None):
        if name is None:
            name = 'Union({}, {})'.format(parent_1.name(), parent_2.name())
        self._parent_1 = parent_1
        self._parent_2 = parent_2

        unique_indices = np.union1d(parent_1.indices(), parent_1.indices())
        self._idcs = unique_indices

        # Use parent_1 as the legal parent
        super().__init__(name, parent_1, None, hidden=hidden)

    def indices(self):
        return self._idcs.copy()

    def parent(self):
        return self._parent_1

    def data(self):
        return np.concatenate((self._parent_1.data(), self._parent_2.data()), axis=0)

    def root(self):
        assert(self._parent_1.root() == self._parent_2.root())
        return self.parent().root()

    def root_data(self):
        return self.root().data()[self.indices(), :]

    def destroy(self):
        del self._idcs
        if self._parent is not None:
            self._parent.remove_child(self)

class RootSelection(Selection):

    def __init__(self, selection, hidden=None):
        if hidden is None:
            hidden = selection.hidden_features()
        parent = selection.root()
        idcs = selection.indices()
        super().__init__(name='RS({})'.format(selection.name()), parent=parent, idcs=idcs, hidden=hidden)