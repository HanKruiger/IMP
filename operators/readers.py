from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model.dataset import Dataset, InputDataset
from operators.operator import Operator

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
            print('Hidden features for {}: {}'.format(name, hidden_features))
            dataset = InputDataset(name, Y, hidden=len(hidden_features))
            out_datasets.append(dataset)

        # Auto-HorzCat labels (may have unintended side-effects, but w/e. Yay for faster testing.)
        if len(out_datasets) == 2 and any([dataset.m == 1 for dataset in out_datasets]) and out_datasets[0].N == out_datasets[1].N:
            if out_datasets[1].m == 1:
                labels = 1
                data = 0
            else:
                labels = 0
                data = 1

            Y = np.column_stack([out_datasets[data].data(), out_datasets[labels].data()])
            merged_dataset = InputDataset(out_datasets[data].name(), Y, hidden=1)
            out_datasets = [merged_dataset]

        self.set_outputs(out_datasets)

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
