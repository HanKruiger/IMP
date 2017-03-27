from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np
from scipy.spatial.distance import cdist

from model import *


class DatasetView:

    def __init__(self, previous=None):
        self._representatives = set()
        self._regulars = set()
        self._vao = dict()
        self._vbo = dict()

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

    def add_regular(self, dataset):
        assert(dataset.is_ready())
        self._regulars.add(dataset)

    def add_representative(self, dataset):
        assert(dataset.is_ready())
        self._representatives.add(dataset)

    def representatives(self):
        assert(len(self._representatives) == 1)
        return list(self._representatives)[0]

    def regulars(self):
        assert(len(self._regulars) == 1)
        return list(self._regulars)[0]

    def datasets(self):
        return set.union(self._regulars, self._representatives)

    def union(self):
        try:
            return self._union
        except AttributeError:
            datasets = list(self.datasets())
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
        assert(all([dataset.n_dimensions() >= 2 for dataset in datasets]))

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

    def make_vbo(self, data, normalize=False, dtype=np.float32):
        X_32 = np.atleast_2d(np.array(data, dtype=dtype))

        if normalize:
            for dim in range(X_32.shape[1]):
                X_32[:, dim] -= X_32[:, dim].min()
                X_32[:, dim] /= X_32[:, dim].max()

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

    def enable(self, shader_program, gl):
        self.shader_program = shader_program

        for dataset in self.datasets():
            self._vao[dataset] = self.make_vao()

            self._vbo[(dataset, 'v_position')] = self.make_vbo(dataset.data()[:, :2])
            self._vbo[(dataset, 'v_position_new')] = self.make_vbo(np.zeros((dataset.n_points(), 2)))
            self._vbo[(dataset, 'v_color')] = self.make_vbo(dataset.data()[:, -1])
            self._vbo[(dataset, 'v_has_old')] = self.make_vbo(np.ones(dataset.n_points()), dtype=np.uint32)
            self._vbo[(dataset, 'v_has_new')] = self.make_vbo(np.zeros(dataset.n_points()), dtype=np.uint32)

            self._vao[dataset].bind()

            position_loc = shader_program.attributeLocation('v_position')
            shader_program.enableAttributeArray(position_loc)
            self._vbo[(dataset, 'v_position')].bind()
            shader_program.setAttributeBuffer(
                position_loc,    # Attribute location
                gl.GL_FLOAT,     # Data type of elements
                0,               # Offset
                2,               # Number of components per vertex
                0                # Stride
            )
            self._vbo[(dataset, 'v_position')].release()

            position_new_loc = shader_program.attributeLocation('v_position_new')
            shader_program.enableAttributeArray(position_new_loc)
            self._vbo[(dataset, 'v_position_new')].bind()
            shader_program.setAttributeBuffer(
                position_new_loc,    # Attribute location
                gl.GL_FLOAT,     # Data type of elements
                0,               # Offset
                2,               # Number of components per vertex
                0                # Stride
            )
            self._vbo[(dataset, 'v_position_new')].release()

            has_old_loc = shader_program.attributeLocation('v_has_old')
            shader_program.enableAttributeArray(has_old_loc)
            self._vbo[(dataset, 'v_has_old')].bind()
            shader_program.setAttributeBuffer(
                has_old_loc,    # Attribute location
                gl.GL_UNSIGNED_BYTE,     # Data type of elements
                0,               # Offset
                1,               # Number of components per vertex
                0                # Stride
            )
            self._vbo[(dataset, 'v_has_old')].release()

            has_new_loc = shader_program.attributeLocation('v_has_new')
            shader_program.enableAttributeArray(has_new_loc)
            self._vbo[(dataset, 'v_has_new')].bind()
            shader_program.setAttributeBuffer(
                has_new_loc,    # Attribute location
                gl.GL_UNSIGNED_BYTE,     # Data type of elements
                0,               # Offset
                1,               # Number of components per vertex
                0                # Stride
            )
            self._vbo[(dataset, 'v_has_new')].release()

            color_loc = shader_program.attributeLocation('v_color')
            shader_program.enableAttributeArray(color_loc)
            self._vbo[(dataset, 'v_color')].bind()
            shader_program.setAttributeBuffer(
                color_loc,    # Attribute location
                gl.GL_FLOAT,  # Data type of elements
                0,            # Offset
                1,            # Number of components per vertex
                0             # Stride
            )
            self._vbo[(dataset, 'v_color')].release()

            self._vao[dataset].release()

        representatives = self.representatives()
        for dataset in self._regulars:
            D = cdist(dataset.data_in_root(), representatives.data_in_root())
            dist_from_repr = D.min(axis=1)
            dist_from_repr -= dist_from_repr.min()
            dist_from_repr /= dist_from_repr.max()

            self._vbo[(dataset, 'v_dist_from_repr')] = self.make_vbo(dist_from_repr)

            self._vao[dataset].bind()
            d_from_repr_loc = shader_program.attributeLocation('v_dist_from_repr')
            shader_program.enableAttributeArray(d_from_repr_loc)
            self._vbo[(dataset, 'v_dist_from_repr')].bind()
            # Explain the format of the attribute buffer to the shader.
            shader_program.setAttributeBuffer(
                d_from_repr_loc,   # Attribute location
                gl.GL_FLOAT,       # Data type of elements
                0,                 # Offset
                1,                 # Number of components per vertex
                0                  # Stride
            )
            self._vbo[(dataset, 'v_dist_from_repr')].release()
            self._vao[dataset].release()

        self._is_active = True

    def disable(self):
        for dataset in self._representatives:
            self._vao[dataset].bind()
            self.shader_program.disableAttributeArray('v_position')
            self.shader_program.disableAttributeArray('v_position_new')
            self.shader_program.disableAttributeArray('v_has_new')
            self.shader_program.disableAttributeArray('v_has_old')
            self.shader_program.disableAttributeArray('v_color')
            self._vao[dataset].release()
            self._vbo[(dataset, 'v_position')].destroy()
            self._vbo[(dataset, 'v_position_new')].destroy()
            self._vbo[(dataset, 'v_has_new')].destroy()
            self._vbo[(dataset, 'v_has_old')].destroy()
            self._vbo[(dataset, 'v_color')].destroy()
            self._vao[dataset].destroy()

        for dataset in self._regulars:
            self._vao[dataset].bind()
            self.shader_program.disableAttributeArray('v_position')
            self.shader_program.disableAttributeArray('v_position_new')
            self.shader_program.disableAttributeArray('v_has_new')
            self.shader_program.disableAttributeArray('v_has_old')
            self.shader_program.disableAttributeArray('v_color')
            self.shader_program.disableAttributeArray('v_dist_from_repr')
            self._vao[dataset].release()
            self._vbo[(dataset, 'v_position')].destroy()
            self._vbo[(dataset, 'v_position_new')].destroy()
            self._vbo[(dataset, 'v_has_new')].destroy()
            self._vbo[(dataset, 'v_has_old')].destroy()
            self._vbo[(dataset, 'v_color')].destroy()
            self._vbo[(dataset, 'v_dist_from_repr')].destroy()
            self._vao[dataset].destroy()

        self._is_active = False

    def draw(self, gl):
        self.shader_program.setUniformValue('u_is_representative', 0)
        for dataset in self._regulars:
            self._vao[dataset].bind()
            gl.glDrawArrays(gl.GL_POINTS, 0, dataset.n_points())
            self._vao[dataset].release()

        self.shader_program.setUniformValue('u_is_representative', 1)
        for dataset in self._representatives:
            self._vao[dataset].bind()
            gl.glDrawArrays(gl.GL_POINTS, 0, dataset.n_points())
            self._vao[dataset].release()

    def __iter__(self):
        return iter(self.datasets())
