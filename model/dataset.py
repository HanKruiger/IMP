from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import abc
import numpy as np
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

        try:
            if self.is_ready():
                pass
        except AttributeError:
            if data is None:
                self._is_ready = False
            else:
                self._is_ready = True

        if parent is not None:
            parent.add_child(self)

    def n_points(self):
        if not self.is_ready():
            return None
        return self.data().shape[0]

    def n_dimensions(self, count_hidden=True):
        try:
            n_dims = self._n_dimensions
        except AttributeError:
            if not self.is_ready():
                return None
            if len(self.data().shape) < 2:
                n_dims = 1
            else:
                n_dims = self.data().shape[1]
            pass

        if not count_hidden:
            return n_dims - self._n_hidden_features
        else:
            return n_dims

    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def is_ready(self):
        return self._is_ready

    def destroy(self):
        if self._data is not None:
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

    def data(self, split_hidden=False):
        if split_hidden:
            return (self._data[:, :-self.hidden_features()], self._data[:, -self.hidden_features():])
        else:
            return self._data

    @pyqtSlot(object)
    def set_data(self, data):
        self._data = data
        self._n_points = data.shape[0]
        self._n_dimensions = data.shape[1]
        self._is_ready = True
        self.data_changed()

    def indices_in_root(self):
        if self.parent() is None:
            return np.arange(self.n_points())
        else:
            return self.parent().indices_in_root()[self.indices_in_parent()]

    def data_in_root(self, split_hidden=False):
        return self.root().data(split_hidden=split_hidden)[self.indices_in_root(), :]

    def hidden_features(self):
        return self._n_hidden_features

    def remove_worker(self, worker_thread):
        self._workers.remove(worker_thread)

    def spawn_thread(self, worker, callback, waitfor=None):
        thread = QThread()
        worker.moveToThread(thread)

        # Save references
        self._workers.add(worker)

        thread.started.connect(worker.work)
        worker.ready.connect(thread.quit)
        worker.ready.connect(callback)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Clean up reference to worker when finished
        thread.finished.connect(
            lambda: self.remove_worker(worker)
        )

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

    class Worker(QObject):
        # Must be overwritten if number of objects differs
        ready = pyqtSignal(object)

        def __init__(self):
            super().__init__()

        def moveToThread(self, thread):
            super().moveToThread(thread)
            # Save reference to thread
            self.thread = thread

        def deleteLater(self):
            super().deleteLater()

        @abc.abstractmethod
        def work(self):
            """Method that should do the work. E.g., make an embedding."""

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


class Selection(Dataset):

    def __init__(self, parent, idcs=None, name=None, hidden=None):
        if name is None:
            name = 'S({})'.format(parent.name())
        self._idcs_in_parent = idcs
        if idcs is not None:
            self._is_ready = True
        super().__init__(name, parent, None, hidden=hidden)

    def data(self, split_hidden=False):
        if self.is_ready():
            if split_hidden:
                return (self.parent().data()[self.indices_in_parent(), :-self.hidden_features()], self.parent().data()[self.indices_in_parent(), -self.hidden_features():])
            else:
                return self.parent().data()[self.indices_in_parent(), :]
        else:
            return None

    def root(self):
        return self.parent().root()

    def indices_in_root(self):
        return self.parent().indices_in_root()[self._idcs_in_parent]

    def indices_in_parent(self):
        return self._idcs_in_parent

    @pyqtSlot(object)
    def set_indices_in_parent(self, idcs):
        self._idcs_in_parent = idcs
        self._is_ready = True
        self.data_changed()


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

    class Unioner(Dataset.Worker):

        ready = pyqtSignal()

        def __init__(self):
            super().__init__()

        def work(self):
            self.ready.emit()

    def __init__(self, parent_1, parent_2, name=None, hidden=None, async=True):
        if name is None:
            name = 'Union({}, {})'.format(parent_1.name(), parent_2.name())
        self._parent_1 = parent_1
        self._parent_2 = parent_2

        # Use parent_1 as the legal parent
        super().__init__(name, parent_1, None, hidden=hidden)

        if async:
            unioner = Union.Unioner()
            self.spawn_thread(unioner, self.dependencies_met, waitfor=(parent_1, parent_2))
        else:
            assert(parent_1.is_ready() and parent_2.is_ready())
            self.dependencies_met()

    def indices_in_root(self):
        return np.concatenate((self._parent_1.indices_in_root(), self._parent_2.indices_in_root()), axis=0)

    def indices_in_parent(self):
        return (np.arange(self._parent_1.n_points()), np.arange(self._parent_2.n_points()))

    @pyqtSlot()
    def dependencies_met(self):
        self._is_ready = True
        self.data_changed()

    def data(self, split_hidden=False):
        if self._is_ready:
            if split_hidden:
                return (
                    np.row_stack((self._parent_1.data()[:, :-self.hidden_features()], self._parent_2.data()[:, :-self.hidden_features()])),
                    np.row_stack((self._parent_1.data()[:, -self.hidden_features():], self._parent_2.data()[:, -self.hidden_features():]))
                )
            else:
                return np.row_stack((self._parent_1.data(), self._parent_2.data()))
        else:
            return None

    def root(self):
        assert(self._parent_1.root() == self._parent_2.root())
        return self.parent().root()

class Difference(Selection):


    def __init__(self, parent_1, parent_2, name=None, hidden=None):
        if name is None:
            name = 'Difference({}, {})'.format(parent_1.name(), parent_2.name())
        # Use parent_1 as the legal parent (this is the only one that is indexed!)
        super().__init__(parent_1, name=name, hidden=hidden)

        differencer = Difference.Differencer(parent_1, parent_2)
        self.spawn_thread(differencer, self.set_indices_in_parent, waitfor=(parent_1, parent_2))

    class Differencer(Dataset.Worker):

        def __init__(self, parent_1, parent_2):
            super().__init__()
            self.parent_1 = parent_1
            self.parent_2 = parent_2

        def work(self):
            idcs_in_root = np.setdiff1d(self.parent_1.indices_in_root(), self.parent_2.indices_in_root())
            # parent_1_idcs_s = sorted
            assert(np.all(self.parent_1.indices_in_root() == np.sort(self.parent_1.indices_in_root())))
            idcs_in_parent = np.searchsorted(self.parent_1.indices_in_root(), idcs_in_root)
            self.ready.emit(idcs_in_parent)


class RootSelection(Selection):

    def __init__(self, selection, hidden=None):
        if hidden is None:
            hidden = selection.hidden_features()
        parent = selection.root()
        idcs = selection.indices_in_root()
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
        if all([dataset.is_ready() for dataset in self.datasets]):
            self.ready.emit()

    def is_ready(self):
        return all([dataset.is_ready() for dataset in self.datasets])

