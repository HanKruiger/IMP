from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *

import abc
import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


class InputDataset(Dataset):

    class Reader(Dataset.Worker):

        def __init__(self, paths):
            super().__init__()
            self.paths = paths

        def work(self):
            out_datasets = []
            for path in self.paths:
                X, hidden_features = self.read(path)
                name = os.path.basename(path).split('.')[0]
                out_datasets.append(X)
            X_full = np.column_stack(out_datasets)

            self.ready.emit(X_full)
        
        def read(self, path):
            hidden_features = []
            if path.endswith('.nd') or path.endswith('.2d'):
                try:
                    X, hidden_features = self.read_nd(path)
                    return X, hidden_features
                except:
                    pass # Hope that numpy will read it..
            X = np.loadtxt(path)
            
            # Normalize single-dimensional datasets (these are often, if not always, labels)
            if X.ndim == 1:
                X /= X.max()

            return X, hidden_features

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
                
                return X, hidden_features

    def __init__(self, paths):
        super().__init__(os.path.basename(paths[0]).split('.')[0], None, None, hidden=1)

        reader = InputDataset.Reader(paths)
        self.spawn_thread(reader, self.set_data)

    def root(self):
        return self