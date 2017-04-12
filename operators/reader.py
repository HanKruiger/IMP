from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

from model import Dataset
from operators import Operator


class Reader(Operator):

    ready = pyqtSignal(object, object)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def work(self):
        X, labels = self.read(self.path)

        # dataset = Dataset(X, np.arange(X.shape[0]), name='input', is_root=True)

        self.ready.emit(X, labels)

    def read(self, path):
        if path.endswith('.nd') or path.endswith('.2d'):
            # try:
            X, labels = self.read_nd(path)
            return X, labels
            # except:
                # pass  # Hope that numpy will read it..
        X = np.loadtxt(path)

        # Normalize single-dimensional datasets (these are often, if not always, labels)
        if X.ndim == 1:
            X /= X.max()

        return X, None

    def read_nd(self, path):
        with open(path) as f:
            all_lines = f.read().splitlines()

            it = iter(all_lines)

            # Ignore preamble
            line = next(it).strip()
            assert(line == 'DY')

            line = next(it).strip()
            N = int(line)

            line = next(it).strip()
            m_nonhidden = int(line)

            line = next(it).strip()
            while line == '':
                line = next(it).strip()

            m = len(line.split(';')) - 1  # First entry is label which we'll ignore for now.
            m_hidden = m - m_nonhidden

            hidden_features = np.arange(m_nonhidden, m_nonhidden + m_hidden)

            X = np.zeros((N, m))
            try:
                for i in range(N):
                    entries = line.split(';')
                    assert(len(entries) >= m_nonhidden + 1)
                    X[i, :] = np.array([float(entry) for entry in entries[1:]])
                    line = next(it)
                print('WARNING: There is trailing data in {}!'.format(path))
            except StopIteration:
                pass

            # Normalize the hidden features (labels) between 0 and 1.
            X[:, hidden_features] /= X[:, hidden_features].max(axis=0)

            labels = X[:, hidden_features]
            X = np.delete(X, hidden_features, axis=1)

            if labels.shape[1] == 1:
                labels = labels.flatten()

            return X, labels
