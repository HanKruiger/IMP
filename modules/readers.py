from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import Dataset, InputDataset
from modules.operator import Operator

import abc
import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


class Reader(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        paths = self.parameters()['paths']

        out_datasets = []
        for path in paths:
            Y, hidden_features = self.read(path)
            name = os.path.basename(path).split('.')[0]
            dataset = InputDataset(name, Y, hidden=len(hidden_features))
            out_datasets.append(dataset)

        self.set_outputs(out_datasets)

    def read(self, path):
        hidden_features = []
        if path.endswith('.nd') or path.endswith('.2d'):
            try:
                X, hidden_features = self.read_nd(path)
            except:
                pass # Hope that numpy will read it..
        X = np.loadtxt(path)

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

            return X, hidden_features

    def set_input(self):
        pass

    @classmethod
    def input_description(cls):
        return {}

    @classmethod
    def parameter_description(cls):
        return {
            'paths': (list, str, [''])
        }
