from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np


class Dataset(QObject):

    # Emitted when (child) embedding is finished
    embedding_finished = pyqtSignal(object)

    def __init__(self, name, parent=None, relation='root', X=None, item_data=None, support=None):
        super().__init__()
        self.name = name
        self._parent = parent
        self.relation = relation

        # Not very elegant. Maybe replace with operator class
        if relation in ['kmeans', 'mb_kmeans']:
            self._is_clustering = True
        else:
            self._is_clustering = False
        print(self._is_clustering)

        self._children = []
        self._item_data = item_data
        self._q_item = None
        self._vbo = None
        self._support = support
        if X is not None:
            self.set_data(X)

    def destroy(self):
        del self.X
        if self._parent is not None:
            self._parent.remove_child(self)
        self.destroy_vbo()

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

    def support(self):
        return self._support

    def is_clustering(self):
        return self._is_clustering

    def set_data(self, X):
        self.X = X
        self.N = X.shape[0]
        if len(X.shape) < 2:
            self.m = 1
        else:
            self.m = X.shape[1]

    def make_embedding(self, embedder):
        self.embedding_worker = embedder
        embedder.set_input(self)
        embedder.finished.connect(self.set_embedding)
        embedder.start()

    @pyqtSlot()
    def set_embedding(self):
        # Fetch data from worker, and delete it
        new_child = self.embedding_worker.out_dataset
        del self.embedding_worker  # Your services are no longer needed.

        self.append_child(new_child)
        self.embedding_finished.emit(new_child)

    def vbo(self):
        if self._vbo is None:
            self.make_vbo()
        return self._vbo

    def make_vbo(self):
        X_32 = np.array(self.X, dtype=np.float32)
        X_32 /= X_32.max()  # Normalize for now.

        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbo.create()
        self._vbo.bind()
        self._vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self._vbo.allocate(X_32.data, X_32.data.nbytes)
        self._vbo.release()

    def destroy_vbo(self):
        if self._vbo is not None:
            self._vbo.destroy()
            self._vbo = None


class InputDataset(Dataset):

    # Emitted when input data is loaded
    data_ready = pyqtSignal()

    def __init__(self, path):
        name = os.path.basename(path).split('.')[0]
        super().__init__(name)

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
