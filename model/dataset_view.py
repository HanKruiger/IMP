from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np
from scipy.spatial.distance import cdist

from model import *


class DatasetView:

    def __init__(self, previous=None):
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

    def set_regular(self, dataset):
        assert(dataset.is_ready())
        self._old_regular = dataset

    def set_new_regular(self, dataset):
        assert(dataset.is_ready())
        self._new_regular = dataset

    def set_representative(self, dataset):
        assert(dataset.is_ready())
        self._old_representative = dataset

    def set_new_representative(self, dataset):
        assert(dataset.is_ready())
        self._new_representative = dataset

    def representative(self):
        return self._new_representative

    def regular(self):
        return self._new_regular

    def datasets(self, new=True, old=False):
        if new and not old:
            return [self._new_regular, self._new_representative]
        elif new and old:
            return [self._old_regular, self._new_regular, self._old_representative, self._new_representative]
        elif old:
            return [self._old_regular, self._old_representative]

    def union(self, new=True, old=False):
        try:
            return self._union
        except AttributeError:
            datasets = self.datasets(new=new, old=old)
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

    def make_vbo(self, data, normalize=False):
        data = np.atleast_2d(data.copy())

        if normalize:
            for dim in range(data.shape[1]):
                data[:, dim] -= data[:, dim].min()
                data[:, dim] /= data[:, dim].max()

        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vbo.allocate(data.data, data.data.nbytes)
        vbo.release()

        return vbo

    def make_vao(self):
        vao = QOpenGLVertexArrayObject()
        vao.create()
        return vao

    def enable(self, shader_program, gl):
        self.shader_program = shader_program

        try:
            root_idcs = np.sort(np.unique(np.concatenate([
                self._old_regular.indices_in_root(),
                self._new_regular.indices_in_root(),
                self._old_representative.indices_in_root(),
                self._new_representative.indices_in_root()
            ])))
            N = root_idcs.size
            
            old_repr_indices = self._old_representative.indices_from_root(root_idcs)
            new_repr_indices = self._new_representative.indices_from_root(root_idcs)
            old_regular_indices = self._old_regular.indices_from_root(root_idcs)
            new_regular_indices = self._new_regular.indices_from_root(root_idcs)
            
            v_has_old = np.array(np.logical_or(old_repr_indices != -1, old_regular_indices != -1), dtype=np.ubyte)
            v_has_new = np.array(np.logical_or(new_repr_indices != -1, new_regular_indices != -1), dtype=np.ubyte)
            
            v_position_old = np.zeros((N, 2), dtype=np.float32)
            v_position_old[old_repr_indices != -1, :] = self._old_representative.data()[np.delete(old_repr_indices, np.where(old_repr_indices == -1)), :2]
            v_position_old[old_regular_indices != -1, :] = self._old_regular.data()[np.delete(old_regular_indices, np.where(old_regular_indices == -1)), :2]
            
            v_position_new = np.zeros((N, 2), dtype=np.float32)
            v_position_new[new_repr_indices != -1, :] = self._new_representative.data()[np.delete(new_repr_indices, np.where(new_repr_indices == -1)), :2]
            v_position_new[new_regular_indices != -1, :] = self._new_regular.data()[np.delete(new_regular_indices, np.where(new_regular_indices == -1)), :2]
            
            v_color = np.zeros(N, dtype=np.float32)
            v_color[old_repr_indices != -1] = self._old_representative.data()[np.delete(old_repr_indices, np.where(old_repr_indices == -1)), -1]
            v_color[new_repr_indices != -1] = self._new_representative.data()[np.delete(new_repr_indices, np.where(new_repr_indices == -1)), -1]
            v_color[old_regular_indices != -1] = self._old_regular.data()[np.delete(old_regular_indices, np.where(old_regular_indices == -1)), -1]
            v_color[new_regular_indices != -1] = self._new_regular.data()[np.delete(new_regular_indices, np.where(new_regular_indices == -1)), -1]

            v_is_repr = np.array(old_repr_indices != -1, dtype=np.ubyte)
            v_is_repr_new = np.array(new_repr_indices != -1, dtype=np.ubyte)

            D_old = cdist(self._old_regular.data_in_root()[np.delete(old_regular_indices, np.where(old_regular_indices == -1)), :], self._old_representative.data_in_root()[np.delete(old_repr_indices, np.where(old_repr_indices == -1)), :])
            dist_from_repr_old = D_old.min(axis=1)
            dist_from_repr_old -= dist_from_repr_old.min()
            dist_from_repr_old /= dist_from_repr_old.max()

            v_dist_from_repr_old = np.zeros(N, dtype=np.float32)
            v_dist_from_repr_old[old_regular_indices != -1] = np.array(dist_from_repr_old, dtype=np.float32)

            D_new = cdist(self._new_regular.data_in_root()[np.delete(new_regular_indices, np.where(new_regular_indices == -1)), :], self._new_representative.data_in_root()[np.delete(new_repr_indices, np.where(new_repr_indices == -1)), :])
            dist_from_repr_new = D_new.min(axis=1)
            dist_from_repr_new -= dist_from_repr_new.min()
            dist_from_repr_new /= dist_from_repr_new.max()

            v_dist_from_repr_new = np.zeros(N, dtype=np.float32)
            v_dist_from_repr_new[new_regular_indices != -1] = np.array(dist_from_repr_new, dtype=np.float32)

        except AttributeError:
            N = self._new_regular.n_points() + self._new_representative.n_points()
            v_position_old = np.zeros((N, 2), dtype=np.float32)
            v_position_new = np.array(np.concatenate([
                self._new_regular.data()[:, :2],
                self._new_representative.data()[:, :2]
            ], axis=0), dtype=np.float32)
            v_color = np.array(np.concatenate([
                self._new_regular.data()[:, -1],
                self._new_representative.data()[:, -1]
            ]), dtype=np.float32)
            v_has_old = np.zeros(N, dtype=np.ubyte)
            v_has_new = np.ones(N, dtype=np.ubyte)
            v_is_repr = np.array(np.concatenate([np.zeros(self._new_regular.n_points()), np.ones(self._new_representative.n_points())]), dtype=np.ubyte)
            v_is_repr_new = np.array(np.concatenate([np.zeros(self._new_regular.n_points()), np.ones(self._new_representative.n_points())]), dtype=np.ubyte)

            v_dist_from_repr_old = np.zeros(N, dtype=np.float32)

            D_new = cdist(self._new_regular.data_in_root(), self._new_representative.data_in_root())
            dist_from_repr_new = D_new.min(axis=1)
            dist_from_repr_new -= dist_from_repr_new.min()
            dist_from_repr_new /= dist_from_repr_new.max()

            v_dist_from_repr_new = np.array(np.concatenate([
                dist_from_repr_new,
                np.zeros(self._new_representative.n_points())
            ]), dtype=np.float32)

        self._n_points = N

        self._vao = self.make_vao()

        self._vbo['v_position_old'] = self.make_vbo(v_position_old)
        self._vbo['v_position_new'] = self.make_vbo(v_position_new)
        self._vbo['v_color'] = self.make_vbo(v_color)
        self._vbo['v_has_old'] = self.make_vbo(v_has_old)
        self._vbo['v_has_new'] = self.make_vbo(v_has_new)
        self._vbo['v_is_repr'] = self.make_vbo(v_is_repr)
        self._vbo['v_is_repr_new'] = self.make_vbo(v_is_repr_new)
        self._vbo['v_dist_from_repr_old'] = self.make_vbo(v_dist_from_repr_old)
        self._vbo['v_dist_from_repr_new'] = self.make_vbo(v_dist_from_repr_new)

        self._vao.bind()

        position_loc = shader_program.attributeLocation('v_position_old')
        shader_program.enableAttributeArray(position_loc)
        self._vbo['v_position_old'].bind()
        shader_program.setAttributeBuffer(
            position_loc,    # Attribute location
            gl.GL_FLOAT,     # Data type of elements
            0,               # Offset
            2,               # Number of components per vertex
            0                # Stride
        )
        self._vbo['v_position_old'].release()

        position_new_loc = shader_program.attributeLocation('v_position_new')
        shader_program.enableAttributeArray(position_new_loc)
        self._vbo['v_position_new'].bind()
        shader_program.setAttributeBuffer(
            position_new_loc,    # Attribute location
            gl.GL_FLOAT,     # Data type of elements
            0,               # Offset
            2,               # Number of components per vertex
            0                # Stride
        )
        self._vbo['v_position_new'].release()

        color_loc = shader_program.attributeLocation('v_color')
        shader_program.enableAttributeArray(color_loc)
        self._vbo['v_color'].bind()
        shader_program.setAttributeBuffer(
            color_loc,    # Attribute location
            gl.GL_FLOAT,  # Data type of elements
            0,            # Offset
            1,            # Number of components per vertex
            0             # Stride
        )
        self._vbo['v_color'].release()

        has_old_loc = shader_program.attributeLocation('v_has_old')
        shader_program.enableAttributeArray(has_old_loc)
        self._vbo['v_has_old'].bind()
        shader_program.setAttributeBuffer(
            has_old_loc,    # Attribute location
            gl.GL_UNSIGNED_BYTE,     # Data type of elements
            0,               # Offset
            1,               # Number of components per vertex
            0                # Stride
        )
        self._vbo['v_has_old'].release()

        has_new_loc = shader_program.attributeLocation('v_has_new')
        shader_program.enableAttributeArray(has_new_loc)
        self._vbo['v_has_new'].bind()
        shader_program.setAttributeBuffer(
            has_new_loc,    # Attribute location
            gl.GL_UNSIGNED_BYTE,     # Data type of elements
            0,               # Offset
            1,               # Number of components per vertex
            0                # Stride
        )
        self._vbo['v_has_new'].release()

        is_repr_loc = shader_program.attributeLocation('v_is_repr')
        shader_program.enableAttributeArray(is_repr_loc)
        self._vbo['v_is_repr'].bind()
        shader_program.setAttributeBuffer(
            is_repr_loc,    # Attribute location
            gl.GL_UNSIGNED_BYTE,     # Data type of elements
            0,               # Offset
            1,               # Number of components per vertex
            0                # Stride
        )
        self._vbo['v_is_repr'].release()

        is_repr_new_loc = shader_program.attributeLocation('v_is_repr_new')
        shader_program.enableAttributeArray(is_repr_new_loc)
        self._vbo['v_is_repr_new'].bind()
        shader_program.setAttributeBuffer(
            is_repr_new_loc,    # Attribute location
            gl.GL_UNSIGNED_BYTE,     # Data type of elements
            0,               # Offset
            1,               # Number of components per vertex
            0                # Stride
        )
        self._vbo['v_is_repr_new'].release()

        dist_from_repr_old_loc = shader_program.attributeLocation('v_dist_from_repr_old')
        shader_program.enableAttributeArray(dist_from_repr_old_loc)
        self._vbo['v_dist_from_repr_old'].bind()
        # Explain the format of the attribute buffer to the shader.
        shader_program.setAttributeBuffer(
            dist_from_repr_old_loc,   # Attribute location
            gl.GL_FLOAT,       # Data type of elements
            0,                 # Offset
            1,                 # Number of components per vertex
            0                  # Stride
        )
        self._vbo['v_dist_from_repr_old'].release()

        dist_from_repr_new_loc = shader_program.attributeLocation('v_dist_from_repr_new')
        shader_program.enableAttributeArray(dist_from_repr_new_loc)
        self._vbo['v_dist_from_repr_new'].bind()
        # Explain the format of the attribute buffer to the shader.
        shader_program.setAttributeBuffer(
            dist_from_repr_new_loc,   # Attribute location
            gl.GL_FLOAT,       # Data type of elements
            0,                 # Offset
            1,                 # Number of components per vertex
            0                  # Stride
        )
        self._vbo['v_dist_from_repr_new'].release()
        
        self._vao.release()

        self._is_active = True

    def disable(self):
        self._vao.bind()
        self.shader_program.disableAttributeArray('v_position_old')
        self.shader_program.disableAttributeArray('v_position_new')
        self.shader_program.disableAttributeArray('v_color')
        self.shader_program.disableAttributeArray('v_has_old')
        self.shader_program.disableAttributeArray('v_has_new')
        self.shader_program.disableAttributeArray('v_is_repr')
        self.shader_program.disableAttributeArray('v_is_repr_new')
        self.shader_program.disableAttributeArray('v_dist_from_repr_old')
        self.shader_program.disableAttributeArray('v_dist_from_repr_new')
        self._vao.release()

        self._vbo['v_position_old'].destroy()
        self._vbo['v_position_new'].destroy()
        self._vbo['v_color'].destroy()
        self._vbo['v_has_old'].destroy()
        self._vbo['v_has_new'].destroy()
        self._vbo['v_is_repr'].destroy()
        self._vbo['v_is_repr_new'].destroy()
        self._vbo['v_dist_from_repr_old'].destroy()
        self._vbo['v_dist_from_repr_new'].destroy()

        self._vao.destroy()

        self._is_active = False

    def draw(self, gl):
        self._vao.bind()
        gl.glDrawArrays(gl.GL_POINTS, 0, self._n_points)
        self._vao.release()

    def __iter__(self):
        return iter(self.datasets())
