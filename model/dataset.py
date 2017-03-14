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

        if data is None:
            self._is_ready = False
        else:
            self._is_ready = True

        if parent is not None:
            parent.add_child(self)

    def n_points(self):
        if self.data() is None:
            return None
        return self.data().shape[0]

    def n_dimensions(self):
        if self.data() is None:
            return None
        if len(self.data().shape) < 2:
            return 1
        else:
            return self.data().shape[1]

    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def is_ready(self):
        return self._is_ready

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
        print('Emitting {}'.format(self.data_ready))
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
        self._is_ready = True
        self.data_changed()

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

        if waitfor is not None:
            waitfor = MultiWait(waitfor)
            if waitfor.is_ready():
                thread.start()
            else:
                thread.waitfor = waitfor  # Thread keeps reference to where it waits on.
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


class Clustering(Dataset):

    def __init__(self, name, parent, X, support, hidden=None):
        super().__init__(name, parent, X, hidden=hidden)
        self._support = support

    def support(self):
        return self._support


class Selection(Dataset):

    def __init__(self, parent, idcs=None, name=None, hidden=None):
        if name is None:
            name = 'S({})'.format(parent.name())
        self._idcs = idcs
        super().__init__(name, parent, None, hidden=hidden)
        if idcs is not None:
            self._is_ready = True

    def data(self):
        return self.parent().data()[self._idcs, :]

    def root(self):
        return self.parent().root()

    def indices(self):
        return self.parent().indices()[self._idcs]

    def destroy(self):
        del self._idcs
        if self._parent is not None:
            self._parent.remove_child(self)

    @pyqtSlot(object)
    def set_indices(self, idcs):
        self._idcs = idcs
        self._is_ready = True
        self.data_changed()


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

        self.spawn_thread(fetcher, self.set_indices, waitfor=(query_nd, root))


class Merging(Dataset):

    def __init__(self, parent_1, parent_2, add_as_hidden=True, name=None, hidden=None):
        if name is None:
            name = 'M({}, {})'.format(parent_1.name(), parent_2.name())
        if add_as_hidden:
            hidden = parent_2.n_dimensions()
        else:
            raise NotImplementedError

        X = np.column_stack((parent_1.data(), parent_2.data()))
        super().__init__(name, parent_1, X, hidden=hidden)

    def root(self):
        return self

class Union(Dataset):

    class Unioner(QObject):

        ready = pyqtSignal()

        def __init__(self, parent_1, parent_2):
            super().__init__()
            self.parent_1 = parent_1
            self.parent_2 = parent_2

        def work(self):
            self.ready.emit()

    def __init__(self, parent_1, parent_2, name=None, hidden=None):
        if name is None:
            name = 'Union({}, {})'.format(parent_1.name(), parent_2.name())
        self._parent_1 = parent_1
        self._parent_2 = parent_2

        # Use parent_1 as the legal parent
        super().__init__(name, parent_1, None, hidden=hidden)

        unioner = Union.Unioner(parent_1, parent_2)
        self.spawn_thread(unioner, self.set_indices, waitfor=(parent_1, parent_2))

    def indices(self):
        return np.concatenate((self._parent_1.indices(), self._parent_2.indices()), axis=0)

    @pyqtSlot()
    def set_indices(self):
        self._is_ready = True
        self.data_changed()

    def data(self):
        if self._is_ready:
            return np.row_stack((self._parent_1.data(), self._parent_2.data()))
        else:
            return None

    def root(self):
        assert(self._parent_1.root() == self._parent_2.root())
        return self.parent().root()

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


class MultiWait(QObject):
    ready = pyqtSignal()

    def __init__(self, datasets):
        super().__init__()
        self.datasets = set(datasets)
        for dataset in datasets:
            if not dataset.is_ready():
                dataset.data_ready.connect(self.subset_is_ready)

    @pyqtSlot(object)
    def subset_is_ready(self, ready_dataset):
        print('{} is ready!'.format(ready_dataset))
        if all([dataset.is_ready() for dataset in self.datasets]):
            self.ready.emit()

    def is_ready(self):
        return all([dataset.is_ready() for dataset in self.datasets])

class Worker(QObject):
    # Must be overwritten of number of objects differs
    ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def work(self):
        """Method that should do the work. E.g., make an embedding."""