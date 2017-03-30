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
        try:
            return self._n_points
        except AttributeError:
            if not self.is_ready():
                return None
            return self.data().shape[0]

    def n_dimensions(self, count_hidden=True):
        if not self.is_ready():
            return None
        if len(self.data().shape) < 2:
            n_dims = 1
        else:
            n_dims = self.data().shape[1]

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

    def data_changed(self):
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

    def root_indexed_data(self, root_indices):
        indices = self.indices_from_root(root_indices)
        return self.data()[indices, :]

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

    def indices_from_root(self, root_idcs):
        try:
            return np.array([self._root_idcs_lookup[i] for i in root_idcs])
        except AttributeError:
            # assert(np.all(self.indices_in_root() == np.sort(self.indices_in_root())))
            self._root_idcs_lookup = defaultdict(lambda: -1)
            for i, idx in enumerate(self.indices_in_root()):
                self._root_idcs_lookup[idx] = i
            return self.indices_from_root(root_idcs)

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

        thread.started.connect(worker.start)
        worker.really_ready.connect(thread.quit)
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
        really_ready = pyqtSignal()

        def __init__(self):
            super().__init__()

        def moveToThread(self, thread):
            super().moveToThread(thread)
            # Save reference to thread
            self.thread = thread

        def deleteLater(self):
            super().deleteLater()

        def start(self):
            t_0 = time.time()
            self.work()
            t_diff = time.time() - t_0
            if t_diff > .1:
                print('{} took {:.2f} s.'.format(self.__class__.__name__, t_diff))
            self.really_ready.emit()

        @abc.abstractmethod
        def work(self):
            """Method that should do the work. E.g., make an embedding."""


class Selection(Dataset):

    def __init__(self, parent, idcs=None, name=None, hidden=None):
        if name is None:
            name = 'S({})'.format(parent.name())
        
        self._n_dimensions = parent.n_dimensions()
        if idcs is not None:
            self._idcs_in_parent = idcs
            self._n_points = idcs.size
            self._is_ready = True
        super().__init__(name, parent, None, hidden=hidden)

    def n_dimensions(self, count_hidden=True):
        if not count_hidden:
            return self._n_dimensions - self.hidden_features()
        return self._n_dimensions

    def data(self, split_hidden=False):
        if self.is_ready():
            if split_hidden:
                non_hidden = self.parent().data()[self.indices_in_parent(), :-self.hidden_features()]
                hidden = self.parent().data()[self.indices_in_parent(), -self.hidden_features():]
                return (non_hidden, hidden)
            else:
                return self.parent().data()[self.indices_in_parent(), :]
        else:
            return None

    def root(self):
        return self.parent().root()

    def indices_in_root(self):
        return self.parent().indices_in_root()[self.indices_in_parent()]

    def indices_in_parent(self):
        return self._idcs_in_parent

    @pyqtSlot(object)
    def set_indices_in_parent(self, idcs):
        self._idcs_in_parent = idcs
        self._n_points = idcs.size
        self._n_dimensions = self.parent().n_dimensions()
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

    def __init__(self, parent_1, parent_2, name=None, hidden=None, async=True):
        if name is None:
            name = 'Union({}, {})'.format(parent_1.name(), parent_2.name())
        self._parent_1 = parent_1
        self._parent_2 = parent_2

        assert(parent_1.n_dimensions() == parent_2.n_dimensions())
        self._n_dimensions = parent_1.n_dimensions()

        # Use parent_1 as the legal parent
        super().__init__(name, parent_1, None, hidden=hidden)

        unioner = Union.Unioner(parent_1, parent_2)
        if async:
            self.spawn_thread(unioner, self.set_indices_in_root_and_parents, waitfor=(parent_1, parent_2))
        else:
            assert(parent_1.is_ready() and parent_2.is_ready())
            root_idcs, indices_in_parent = unioner.work(async=False)
            self.set_indices_in_root_and_parents(root_idcs, indices_in_parent)
    
    class Unioner(Dataset.Worker):

        ready = pyqtSignal(object, object)

        def __init__(self, parent_1, parent_2):
            super().__init__()
            self.parent_1 = parent_1
            self.parent_2 = parent_2

        def work(self, async=True):
            # indices_in_parent: First column: Indicates which parent. Second column: Index in that parent.
            indices_in_parent = np.concatenate([
                np.full((self.parent_1.n_points(), 2), 0),
                np.full((self.parent_2.n_points(), 2), 1)
            ])
            indices_in_parent[:self.parent_1.n_points(), 1] = np.arange(self.parent_1.n_points())
            indices_in_parent[self.parent_1.n_points():, 1] = np.arange(self.parent_2.n_points())
            root_idcs = np.concatenate([
                self.parent_1.indices_in_root(), self.parent_2.indices_in_root()
            ])
            
            # Sort rows of indices_in_parent such that the root indices are sorted.
            sorted_idcs = np.argsort(root_idcs)
            indices_in_parent = indices_in_parent[sorted_idcs, :]
            if async:
                self.ready.emit(root_idcs[sorted_idcs], indices_in_parent)
            else:
                return root_idcs[sorted_idcs], indices_in_parent

    @pyqtSlot(object, object)
    def set_indices_in_root_and_parents(self, root_idcs, indices_in_parent):
        self._indices_in_root = root_idcs
        self._indices_in_parent = indices_in_parent
        self._n_points = root_idcs.size
        self._is_ready = True
        self.data_changed()

    def indices_in_root(self):
        return self._indices_in_root

    def indices_in_parent(self):
        return self._indices_in_parent

    def data(self, split_hidden=False):
        if self._is_ready:
            which_parent = self.indices_in_parent()[:, 0]
            which_index = self.indices_in_parent()[:, 1]

            data = np.zeros((self.n_points(), self._parent_1.n_dimensions()))
            data[which_parent == 0] = self._parent_1.data()[which_index[which_parent == 0], :]
            data[which_parent == 1] = self._parent_2.data()[which_index[which_parent == 1], :]

            if split_hidden:
                non_hidden = data[:, :-self.hidden_features()]
                hidden = data[:, -self.hidden_features():]
                return (non_hidden, hidden)
            else:
                return data
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
        self._n_points = None
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
            try:
                assert(np.all(self.parent_1.indices_in_root() == np.sort(self.parent_1.indices_in_root())))
            except AssertionError:
                print('Indices in {} are not sorted.'.format(self.parent_1.name()))
            idcs_in_parent = np.searchsorted(self.parent_1.indices_in_root(), idcs_in_root)
            self.ready.emit(idcs_in_parent)


class RootSelection(Selection):

    def __init__(self, selection, hidden=None):
        if hidden is None:
            hidden = selection.hidden_features()
        name = 'RS({})'.format(selection.name())
        self._n_points = selection.n_points()
        super().__init__(selection.root(), idcs=None, name=name, hidden=hidden)

        root_selector = RootSelection.RootSelector(selection)
        self.spawn_thread(root_selector, self.set_indices_in_parent, waitfor=(selection,))

    class RootSelector(Dataset.Worker):

        def __init__(self, selection):
            super().__init__()
            self.selection = selection

        def work(self):
            idcs_in_root = self.selection.indices_in_root()
            self.ready.emit(idcs_in_root)


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

