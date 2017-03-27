from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np
from scipy.spatial.distance import cdist

from model import *


class DatasetView:

    def __init__(self, previous=None):
        self._viewed_datasets = dict()
        self.shader_program = None
        self._previous = previous
        self._is_active = False

    def previous(self):
        self._previous._next = self
        return self._previous

    def next(self):
        assert(self._next._previous == self)
        return self._next

    def is_active(self):
        return self._is_active

    def add_dataset(self, dataset, kind):
        assert(dataset.is_ready())
        self._viewed_datasets[dataset] = {
            'kind': kind,
            'vao': None,
            'vbos': dict()
        }

    def get(self, kind):
        for dataset, viewed_dataset in self._viewed_datasets.items():
            if viewed_dataset['kind'] == kind:
                return dataset
        raise KeyError

    def datasets(self):
        return [dataset for dataset in self._viewed_datasets.keys()]

    def union(self):
        try:
            return self._union
        except AttributeError:
            datasets = self.datasets()
            union = datasets[0]
            for dataset in datasets[1:]:
                union = Union(union, dataset, async=False)
            self._union = union
            return self._union


    def name(self):
        return ', '.join([dataset.name() for dataset in self.datasets()])

    def root(self):
        root = datasets[0].root()
        assert(all([dataset.root == root for dataset in self.datasets()]))
        return root

    def get_bounds(self):
        datasets = self.datasets()
        assert(all([dataset.n_dimensions() == 3 for dataset in datasets]))

        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for dataset in datasets:
            x_min = min(x_min, dataset.data()[:, 0].min())
            x_max = max(x_max, dataset.data()[:, 0].max())
            y_min = min(y_min, dataset.data()[:, 1].min())
            y_max = max(y_max, dataset.data()[:, 1].max())
        return x_min, x_max, y_min, y_max

    def filter_unseen_points(self, projection_view):
        projection_view = np.array(projection_view.data()).reshape((4, 4))

        # Delete z-entries in transformation matrix (it's 2D, not 3D).
        projection_view = np.delete(projection_view, 2, axis=0)
        projection_view = np.delete(projection_view, 2, axis=1)

        # Build union of all datasets in the current DatasetView
        union = self.union()

        N = union.n_points()
        X = union.data()[:, :2]
        X = np.concatenate((X, np.ones((N, 1))), axis=1)

        # Project points to clip space
        Y = X.dot(projection_view)

        Y_visible = np.abs(Y).max(axis=1) <= 1
        visible_idcs = np.where(Y_visible == True)[0]
        invisible_idcs = np.where(Y_visible == False)[0]

        return visible_idcs, invisible_idcs

    def make_vbo(self, data, normalize=False):
        X_32 = np.array(data, dtype=np.float32)

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

    def make_vao(self):
        vao = QOpenGLVertexArrayObject()
        vao.create()
        return vao

    def init_vaos_and_buffers(self):
        representatives = None
        for dataset, viewed_dataset in self._viewed_datasets.items():
            if viewed_dataset['kind'] == 'representatives':
                representatives = dataset
                break
        
        for dataset, viewed_dataset in self._viewed_datasets.items():
            assert(dataset.n_dimensions() == 3) # for now?

            viewed_dataset['vao'] = self.make_vao()

            if viewed_dataset['kind'] == 'regular' and representatives is not None:
                D = cdist(dataset.data_in_root(), representatives.data_in_root())
                d_from_repr = D.min(axis=1)
                d_from_repr -= d_from_repr.min()
                d_from_repr /= d_from_repr.max()
            else:
                d_from_repr = np.zeros(dataset.n_points())

            viewed_dataset['vbos']['position_x'] = self.make_vbo(dataset.data()[:, 0])
            viewed_dataset['vbos']['position_y'] = self.make_vbo(dataset.data()[:, 1])
            viewed_dataset['vbos']['color'] = self.make_vbo(dataset.data()[:, 2])
            viewed_dataset['vbos']['d_from_repr'] = self.make_vbo(d_from_repr)
            
        self._is_active = True

    def enable_attributes(self, shader_program, gl):
        self.shader_program = shader_program

        for dataset, viewed_dataset in self._viewed_datasets.items():
            vao = self.make_vao()
            viewed_dataset['vao'] = vao

            vao.bind()

            for attribute, vbo in viewed_dataset['vbos'].items():
                attrib_loc = shader_program.attributeLocation(attribute)

                shader_program.enableAttributeArray(attrib_loc)
                
                vbo.bind()

                # Explain the format of the attribute buffer to the shader.
                shader_program.setAttributeBuffer(
                    attrib_loc,    # Attribute location
                    gl.GL_FLOAT,       # Data type of elements
                    0,                      # Offset
                    1,                      # Number of components per vertex
                    0                       # Stride
                )

                vbo.release()

            vao.release()
        self._is_active = True

    def draw(self, gl, shader_program):
        for dataset, viewed_dataset in self._viewed_datasets.items():
            if viewed_dataset['kind'] == 'regular':
                shader_program.setUniformValue('observation_type', 0)
            elif viewed_dataset['kind'] == 'representatives':
                shader_program.setUniformValue('observation_type', 1)
            
            vao = viewed_dataset['vao']
            vao.bind()
            gl.glDrawArrays(gl.GL_POINTS, 0, dataset.n_points())
            vao.release()

    def disable_attributes(self):
        for dataset, viewed_dataset in self._viewed_datasets.items():
            vao = viewed_dataset['vao']

            vao.bind()

            for attribute in ['position_x', 'position_y', 'color']:
                self.shader_program.disableAttributeArray(attribute)

            vao.release()

    def disable(self):
        if self.shader_program is not None:
            self.disable_attributes()
        for viewed_dataset in self._viewed_datasets.values():
            for vbo in viewed_dataset['vbos'].values():
                vbo.destroy()
        self._is_active = False

    def __iter__(self):
        return iter(self._viewed_datasets.items())
