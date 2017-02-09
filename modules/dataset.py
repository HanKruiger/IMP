from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np


class Dataset(QObject):

    # Emitted when (child) embedding is finished
    operation_finished = pyqtSignal(object)

    def __init__(self, name, parent, X, hidden=[]):
        super().__init__()
        self._name = name
        self._parent = parent
        self._hidden_features = hidden.copy()

        self._children = []
        self._q_item = None
        self._vbos = dict()

        if X is not None:
            self.set_data(X)

    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def destroy(self):
        del self.X
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

    def set_data(self, X):
        self.X = X
        self.N = X.shape[0]
        if len(X.shape) < 2:
            self.m = 1
        else:
            self.m = X.shape[1]

    def hidden_features(self):
        return self._hidden_features

    def perform_operation(self, operator):
        self.embedding_worker = operator
        operator.finished.connect(self.operator_finished)
        operator.start()

    @pyqtSlot()
    def operator_finished(self):
        # Fetch data from worker, and delete it
        result = self.embedding_worker.output()
        del self.embedding_worker  # Your services are no longer needed.

        if type(result) == tuple:
            for child in result:
                self.append_child(child)
                self.operation_finished.emit(child)
        else:
            self.append_child(result)
            self.operation_finished.emit(result)

    def vbo(self, dim):
        try:
            return self._vbos[dim]
        except KeyError:
            return self.make_vbo(dim)

    def make_vbo(self, dim):
        X_32 = np.array(self.X[:, dim], dtype=np.float32)
        X_32 /= X_32.max()  # Normalize for now.

        self._vbos[dim] = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbos[dim].create()
        self._vbos[dim].bind()
        self._vbos[dim].setUsagePattern(QOpenGLBuffer.StaticDraw)
        self._vbos[dim].allocate(X_32.data, X_32.data.nbytes)
        self._vbos[dim].release()
        return self._vbos[dim]

    def destroy_vbos(self):
        for dim, vbo in self._vbos.copy().items():
            vbo.destroy()
            del self._vbos[dim]

    def destroy_vbo(self, dim):
        self._vbos[dim].destroy()
        del self._vbos[dim]


class InputDataset(Dataset):

    # Emitted when input data is loaded
    data_ready = pyqtSignal()

    def __init__(self, path):
        name = os.path.basename(path).split('.')[0]
        super().__init__(name, None, None)

        # Load the data in a separate thread, so the GUI doesn't hang.
        # (Doesn't start reading yet. Still needs to be connected from
        # outside.)
        self.worker = DataLoadWorker(path)

        # Somehow I need this function, and cannot call the method directly...
        def wrapper():
            self.on_data_loaded()
        self.worker.finished.connect(wrapper)

    def load_data(self):
        # Start the thread (calls the run() method in the other thread)
        self.worker.start()

    @pyqtSlot()
    def on_data_loaded(self):
        # Fetch data from worker, and delete it
        X = self.worker.data
        del self.worker  # Your services are no longer needed.

        # Set the data.
        super().set_data(X)

        # Tell that we're ready!
        self.data_ready.emit()


# Worker class that loads the data in a separate thread
class DataLoadWorker(QThread):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        # Make numpy load the data
        self.data = np.loadtxt(self.path)


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


class Clustering(Dataset):

    def __init__(self, name, parent, X, support, hidden=[]):
        super().__init__(name, parent, X, hidden=hidden)
        self._support = support

    def support(self):
        return self._support


# Mostly semantics..
class Embedding(Dataset):

    def __init__(self, name, parent, X, hidden=[]):
        super().__init__(name, parent, X, hidden=hidden)


class Sampling(Dataset):

    def __init__(self, name, parent, X, hidden=[]):
        super().__init__(name, parent, X, hidden=hidden)


class Merging(Dataset):

    def __init__(self, name, parent, X, hidden=[]):
        super().__init__(name, parent, X, hidden=hidden)
