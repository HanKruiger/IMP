from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np


class Dataset(QObject):

    # Emitted when (child) embedding is finished
    operation_finished = pyqtSignal(object, object)

    def __init__(self, name, parent, X, hidden=0):
        super().__init__()
        self._name = name
        self._parent = parent
        self._n_hidden_features = hidden

        self._workers = set()
        self._children = []
        self._q_item = None
        self._vbos = dict()

        if X is not None:
            self._X = X
            self.N = X.shape[0]
            if len(X.shape) < 2:
                self.m = 1
            else:
                self.m = X.shape[1]

    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def destroy(self):
        del self._X
        if self._parent is not None:
            self._parent.remove_child(self)
        self.destroy_vbos()

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

    def append_child(self, child):
        self._children.append(child)

    def remove_child(self, child):
        self._children.remove(child)

    def is_clustering(self):
        return self._is_clustering

    def data(self):
        return self._X

    def indices(self):
        if self.parent() is not None:
            return parent().indices()
        else:
            return np.arange(self.N)

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
                child.parent().append_child(child)
        else:
            result.parent().append_child(result)
        
        self.operation_finished.emit(result, operator)

    def vbo(self, dim):
        try:
            return self._vbos[dim]
        except KeyError:
            return self.make_vbo(dim)

    def make_vbo(self, dim, normalize=False):
        X_32 = np.array(self.data()[:, dim], dtype=np.float32)

        if normalize:
            X_32 -= X_32.min()
            X_32 /= X_32.max()

        self._vbos[dim] = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbos[dim].create()
        self._vbos[dim].bind()
        self._vbos[dim].setUsagePattern(QOpenGLBuffer.StaticDraw)
        self._vbos[dim].allocate(X_32.data, X_32.data.nbytes)
        self._vbos[dim].release()
        return self._vbos[dim]

    def normalized_vbo(self, dim):
        try:
            self.destroy_vbo(dim)
        except KeyError:
            pass
        return self.make_vbo(dim, normalize=True)

    def destroy_vbos(self):
        for dim, vbo in self._vbos.copy().items():
            vbo.destroy()
            del self._vbos[dim]

    def destroy_vbo(self, dim):
        self._vbos[dim].destroy()
        del self._vbos[dim]


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

    def __init__(self, name, X, hidden=0):
        super().__init__(name, None, X, hidden=hidden)

    def root(self):
        return self

    def root_data(self):
        return self.data()


class Clustering(Dataset):

    def __init__(self, name, parent, X, support, hidden=0):
        super().__init__(name, parent, X, hidden=hidden)
        self._support = support

    def support(self):
        return self._support


class Embedding(Dataset):

    def __init__(self, name, parent, X, hidden=0):
        super().__init__(name, parent, X, hidden=hidden)

    def root(self):
        return self.parent().root()

    def root_data(self):
        return self.parent().root_data()

class Selection(Dataset):

    def __init__(self, name, parent, idcs, hidden=0):
        self._idcs = idcs
        self.m = parent.m
        self.N = len(idcs)
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
        self.destroy_vbos()


class Sampling(Selection):

    def __init__(self, name, parent, idcs, hidden=0):
        super().__init__(name, parent, idcs, hidden=hidden)

    def set_support(self, support):
        self._support = support

    def support(self):
        return self._support

class Merging(Dataset):

    def __init__(self, name, parent, X, hidden=0):
        super().__init__(name, parent, X, hidden=hidden)

    def root(self):
        return self

    def root_data(self):
        return self.data()