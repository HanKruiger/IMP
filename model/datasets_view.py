from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np


class DatasetsView:

    def __init__(self):
        self._viewed_datasets = dict()

    def add_dataset(self, dataset, kind, lazy=True):
        self._viewed_datasets[dataset] = {
            'kind': kind,
            'vbos': dict()
        }

        if not lazy:
            assert(dataset.m == 2 or dataset.m == 3)
            for dim in range(dataset.m):
                self._viewed_datasets[dataset]['vbos'][dim] = self.make_vbo(dataset, dim)

    def vbo(self, kind, dim):
        for dataset, viewed_dataset in self._viewed_datasets.items():
            if viewed_dataset['kind'] != kind:
                continue

            if not dim in viewed_dataset['vbos']:
                # Lazy generation of VBOs
                viewed_dataset['vbos'][dim] = self.make_vbo(dataset, dim)

            return viewed_dataset['vbos'][dim]

        raise KeyError('No dataset found with kind \'{}\'.'.format(kind))

    def make_vbo(self, dataset, dim, normalize=False):
        X_32 = np.array(dataset.data()[:, dim], dtype=np.float32)

        if normalize:
            X_32 -= X_32.min()
            X_32 /= X_32.max()

        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vbo.allocate(X_32.data, X_32.data.nbytes)
        vbo.release()

        return vbo

    def destroy(self):
        for viewed_dataset in self._viewed_datasets.values():
            for vbo in viewed_dataset['vbos'].values():
                vbo.destroy()

    def __iter__(self):
        return iter(self._viewed_datasets.items())
