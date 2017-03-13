from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np
from functools import partial
from operators.utils import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Dataset(QObject):

    # Emitted when (child) embedding is finished
    has_new_child = pyqtSignal(object)
    data_ready = pyqtSignal(object)
    ready = pyqtSignal()

    def __init__(self, name, parent, data, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = parent.hidden_features()

        self._name = name
        self._parent = parent
        self._n_hidden_features = hidden

        self._workers = set()
        self._children = []
        self._q_item = None

        self._data = data

        if parent is not None:
            parent.add_child(self)

    def n_points(self):
        if self.data() is None:
            return 'NA'
        return self.data().shape[0]

    def n_dimensions(self):
        if self.data() is None:
            return 'NA'
        if len(self.data().shape) < 2:
            return 1
        else:
            return self.data().shape[1]

    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def destroy(self):
        del self._data
        if self._parent is not None:
            self._parent.remove_child(self)

    def parent(self):
        return self._parent

    def q_item(self):
        return self._q_item

    def set_q_item(self, q_item):
        self._q_item = q_item

    def data_changed(self):
        self.q_item().emitDataChanged()
        self.data_ready.emit(self)
        self.ready.emit()

    def child(self, idx):
        return self._children[idx]

    def child_count(self):
        return len(self._children)

    def add_child(self, child):
        self._children.append(child)
        self.has_new_child.emit(child)

    def remove_child(self, child):
        self._children.remove(child)

    def data(self):
        return self._data

    @pyqtSlot(object)
    def set_data(self, data):
        self._data = data
        self.data_changed()

    def data_is_ready(self):
        if len(self._workers) == 0:
            self.ready.emit()
            return True
        return False

    def indices(self):
        if self.parent() is not None:
            return self.parent().indices()
        else:
            return np.arange(self.n_points())

    def hidden_features(self):
        return self._n_hidden_features

    def spawn_thread(self, worker, callback, waitfor=None):
        thread = QThread()
        worker.moveToThread(thread)

        # Save references
        self._workers.add((worker, thread))

        thread.started.connect(worker.work)
        worker.ready.connect(thread.quit)
        worker.ready.connect(callback)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Clean up references
        thread.finished.connect(partial(self._workers.remove, (worker, thread)))

        if waitfor:
            if waitfor.data_is_ready():
                thread.start()
            else:
                waitfor.ready.connect(thread.start)
        else:
            thread.start()


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


class TSNEEmbedding(Dataset):

    class TSNEEmbedder(QObject):

        ready = pyqtSignal(object)

        def __init__(self, parent, **parameters):
            super().__init__()
            self.parent = parent
            self.parameters = parameters

        def work(self):
            # Hide hidden features
            X_use, X_hidden = hide_features(self.parent.data(), self.parent.hidden_features())
            
            # t-SNE embedding
            print(X_use.shape)
            print(self.parameters)
            tsne = TSNE(**self.parameters)
            Y_use = tsne.fit_transform(X_use)

            # Restore original hidden features
            Y = np.concatenate((Y_use, X_hidden), axis=1)

            self.ready.emit(Y)

    def __init__(self, parent, name=None, hidden=None, **parameters):
        if name is None:
            name = 'E({})'.format(parent.name())
        super().__init__(name, parent, data=None, hidden=hidden)

        embedder = TSNEEmbedding.TSNEEmbedder(parent, **parameters)

        self.spawn_thread(embedder, self.set_data, waitfor=parent)

    def root(self):
        return self.parent().root()

    def root_data(self):
        return self.parent().root_data()


class Selection(Dataset):

    def __init__(self, parent, idcs=None, name=None, hidden=None):
        if name is None:
            name = 'S({})'.format(parent.name())
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

    @pyqtSlot(object)
    def set_indices(self, idcs):
        self._idcs = idcs
        self.data_changed()

class RandomSampling(Selection):

    class RandomSampler(QObject):

        ready = pyqtSignal(object)

        def __init__(self, parent, n_samples):
            super().__init__()
            self.parent = parent
            self.n_samples = n_samples

        def work(self):
            idcs = np.random.choice(self.parent.n_points(), self.n_samples, replace=False)
            self.ready.emit(idcs)

    def __init__(self, parent, n_samples, name=None, hidden=None):
        if name is None:
            name = 'RndS({})'.format(parent.name())

        super().__init__(parent, idcs=None, name=name, hidden=hidden)

        sampler = RandomSampling.RandomSampler(parent, n_samples)
        self.spawn_thread(sampler, self.set_indices, waitfor=parent)
    


class KNNFetching(Selection):

    class KNNFetcher(QObject):

        ready = pyqtSignal(object)

        def __init__(self, query_nd, root, n_points):
            super().__init__()
            self.query_nd = query_nd
            self.root = root
            self.n_points = n_points

        def work(self):
            idcs = knn_fetch(self.query_nd, self.root, self.n_points)
            self.ready.emit(idcs)

    def __init__(self, query_2d, n_points, name=None, hidden=None):
        root = query_2d.root()
        if name is None:
            name = 'F({}, {})'.format(root.name(), query_2d.name())
        if hidden is None:
            hidden = root.hidden_features()
        super().__init__(root, idcs=None, name=name, hidden=hidden)
        
        query_nd = RootSelection(query_2d)

        fetcher = KNNFetching.KNNFetcher(query_nd, root, n_points)
        self.spawn_thread(fetcher, self.set_indices)


class Merging(Dataset):

    def __init__(self, parent_1, parent_2, X, name=None, hidden=None):
        if name is None:
            name = 'M({}, {})'.format(parent_1.name(), parent_2.name())
        if hidden is None:
            hidden = parent_2.n_dimensions()
        super().__init__(name, parent_1, X, hidden=hidden)

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
