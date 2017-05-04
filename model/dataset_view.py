from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np
from scipy.spatial.distance import cdist

from model import *
from utils.buffers import *


class DatasetView:

    def __init__(self, previous=None):
        # self.shader_program = None
        self._vbo = dict()
        self._view_new = QMatrix4x4()

        if previous is not None:
            self._previous = previous
            self._view_old = previous._view_new
        else:
            self._view_old = QMatrix4x4()

        self._is_active = False

    def previous(self):
        # Place this DatasetView as the next step after the previous.
        self._previous._next = self

        # For visual continuity, use the current old view matrix as the previous new matrix.
        self._previous._view_new = self._view_old

        return self._previous

    def next(self):
        assert(self._next._previous == self)
        return self._next

    def is_active(self):
        return self._is_active

    def set_old_regular(self, dataset):
        self._old_regular = dataset

    def set_new_regular(self, dataset):
        self._new_regular = dataset

    def set_old_representative(self, dataset):
        self._old_representative = dataset

    def set_new_representative(self, dataset):
        self._new_representative = dataset

    def root_indices(self):
        return self._root_indices

    def new_representative(self):
        return self._new_representative

    def new_regular(self):
        return self._new_regular

    def old_representative(self):
        return self._old_representative

    def old_regular(self):
        return self._old_regular

    def datasets(self, old_or_new='new'):
        if old_or_new == 'new':
            return [self._new_regular, self._new_representative]
        elif old_or_new == 'old':
            return [self._old_regular, self._old_representative]

    def union(self, old_or_new='new'):
        try:
            if old_or_new == 'new':
                return self._union_new
            elif old_or_new == 'old':
                return self._union_old
        except AttributeError:
            datasets = self.datasets(old_or_new)
            the_union = datasets[0]
            for dataset in datasets[1:]:
                the_union = the_union + dataset

            if old_or_new == 'new':
                self._union_new = the_union
            elif old_or_new == 'old':
                self._union_old = the_union
            return self.union(old_or_new)

    def view_matrix(self, old_or_new='new'):
        if old_or_new == 'new':
            return self._view_new
        elif old_or_new == 'old':
            return self._view_old

    def fit_to_view(self):
        try:
            x_min, x_max, y_min, y_max = self.get_bounds('new')

            # Compute the center of the all-enclosing square, and the distance from center to its sides.
            center = QVector2D((x_min + x_max) / 2, (y_min + y_max) / 2)
            dist_to_sides = 0.5 * max(x_max - x_min, y_max - y_min)

            # To account for points that are almost exactly at the border, we need a small
            # tolerance value. (Can also be seen as (very small) padding.)
            tolerance = .2
            dist_to_sides *= 1 + tolerance

            self.view_matrix('new').setToIdentity()
            self.view_matrix('new').scale(1 / dist_to_sides)
            self.view_matrix('new').translate(-center.x(), -center.y())
        except AttributeError:
            print('Could not fit to \'{} bounds\' since it is not set.'.format('new'))

    def get_bounds(self, old_or_new='new'):
        datasets = self.datasets(old_or_new)
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

    def enable(self, shader_program, gl):
        self.shader_program = shader_program

        try:
            root_indices = np.sort(np.unique(np.concatenate([
                self._old_regular.indices(),
                self._new_regular.indices(),
                self._old_representative.indices(),
                self._new_representative.indices()
            ])))
            N = root_indices.size

            old_repr_indices = self._old_representative.root_indices_to_own(root_indices)
            new_repr_indices = self._new_representative.root_indices_to_own(root_indices)
            old_regular_indices = self._old_regular.root_indices_to_own(root_indices)
            new_regular_indices = self._new_regular.root_indices_to_own(root_indices)

            v_has_old = np.array(np.logical_or(old_repr_indices != -1, old_regular_indices != -1), dtype=np.ubyte)
            v_has_new = np.array(np.logical_or(new_repr_indices != -1, new_regular_indices != -1), dtype=np.ubyte)

            v_position_old = np.zeros((N, 2), dtype=np.float32)
            v_position_old[old_repr_indices != -1, :] = self._old_representative.data()[np.delete(old_repr_indices, np.where(old_repr_indices == -1)), :2]
            v_position_old[old_regular_indices != -1, :] = self._old_regular.data()[np.delete(old_regular_indices, np.where(old_regular_indices == -1)), :2]

            v_position_new = np.zeros((N, 2), dtype=np.float32)
            v_position_new[new_repr_indices != -1, :] = self._new_representative.data()[np.delete(new_repr_indices, np.where(new_repr_indices == -1)), :2]
            v_position_new[new_regular_indices != -1, :] = self._new_regular.data()[np.delete(new_regular_indices, np.where(new_regular_indices == -1)), :2]

            v_is_repr_old = np.array(old_repr_indices != -1, dtype=np.ubyte)
            v_is_repr_new = np.array(new_repr_indices != -1, dtype=np.ubyte)

        except AttributeError:
            root_indices = np.concatenate([self._new_regular.indices(), self._new_representative.indices()])
            N = root_indices.size
            v_position_old = np.zeros((N, 2), dtype=np.float32)
            v_position_new = np.array(np.concatenate([
                self._new_regular.data()[:, :2],
                self._new_representative.data()[:, :2]
            ], axis=0), dtype=np.float32)
            v_has_old = np.zeros(N, dtype=np.ubyte)
            v_has_new = np.ones(N, dtype=np.ubyte)
            v_is_repr_old = np.array(np.concatenate([np.zeros(self._new_regular.n_points()), np.ones(self._new_representative.n_points())]), dtype=np.ubyte)
            v_is_repr_new = np.array(np.concatenate([np.zeros(self._new_regular.n_points()), np.ones(self._new_representative.n_points())]), dtype=np.ubyte)

        try:
            v_colour = np.array(self._colour[root_indices], dtype=np.float32)
        except AttributeError:
            v_colour = np.zeros(N, dtype=np.float32)

        self._root_indices = root_indices

        self._n_points = N

        self.set_vao(make_vao())

        self.set_vbo('v_position_old', make_vbo(v_position_old))
        self.set_vbo('v_position_new', make_vbo(v_position_new))
        self.set_vbo('v_has_old', make_vbo(v_has_old))
        self.set_vbo('v_has_new', make_vbo(v_has_new))
        self.set_vbo('v_is_repr_old', make_vbo(v_is_repr_old))
        self.set_vbo('v_is_repr_new', make_vbo(v_is_repr_new))

        self.enable_vbo_attribute('v_position_old', dtype=gl.GL_FLOAT, tuple_size=2)
        self.enable_vbo_attribute('v_position_new', dtype=gl.GL_FLOAT, tuple_size=2)
        self.enable_vbo_attribute('v_has_old', dtype=gl.GL_UNSIGNED_BYTE, tuple_size=1)
        self.enable_vbo_attribute('v_has_new', dtype=gl.GL_UNSIGNED_BYTE, tuple_size=1)
        self.enable_vbo_attribute('v_is_repr_old', dtype=gl.GL_UNSIGNED_BYTE, tuple_size=1)
        self.enable_vbo_attribute('v_is_repr_new', dtype=gl.GL_UNSIGNED_BYTE, tuple_size=1)

        self._is_active = True

    def vao(self):
        return self._vao

    def set_vao(self, vao):
        self._vao = vao

    def vbo(self, name):
        return self._vbo[name]

    def set_vbo(self, name, vbo):
        self._vbo[name] = vbo

    def enable_vbo_attribute(self, vbo_name, dtype, tuple_size):
        vbo = self.vbo(vbo_name)
        self.vao().bind()

        attribute_loc = self.shader_program.attributeLocation(vbo_name)

        self.shader_program.enableAttributeArray(attribute_loc)
        vbo.bind()
        self.shader_program.setAttributeBuffer(
            attribute_loc,  # Attribute location
            dtype,          # Data type of elements
            0,              # Offset
            tuple_size,     # Number of components per vertex
            0               # Stride
        )
        vbo.release()
        self.vao().release()

    def disable(self):
        self.vao().destroy()

        for vbo_name, vbo in self._vbo.items():
            self.shader_program.disableAttributeArray(vbo_name)
            vbo.destroy()

        self._is_active = False

    def draw(self, gl):
        self.shader_program.setUniformValue('u_view_old', self._view_old)
        self.shader_program.setUniformValue('u_view_new', self._view_new)

        self.vao().bind()
        gl.glDrawArrays(gl.GL_POINTS, 0, self._n_points)
        self.vao().release()
