from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np

from model.dataset import Selection


class DatasetsView:

    def __init__(self, previous=None):
        self._viewed_datasets = dict()
        self.shader_program = None
        self.previous = previous

    def add_dataset(self, dataset, kind):
        self._viewed_datasets[dataset] = {
            'kind': kind,
            'vao': None,
            'vbos': dict()
        }

    def datasets(self):
        return [dataset for dataset in self._viewed_datasets.keys()]

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

        visibles = []
        invisibles = []

        for dataset in self.datasets():
            N = dataset.n_points()
            X = dataset.data()[:, :2]
            X = np.concatenate((X, np.ones((N, 1))), axis=1)
            Y = X.dot(projection_view)

            Y_visible = np.abs(Y).max(axis=1) <= 1
            visible_idcs = np.where(Y_visible == True)[0]
            invisible_idcs = np.where(Y_visible == False)[0]

            # Only make new dataset if it's non-empty.
            if visible_idcs.shape[0] > 0:
                visible_selection = Selection(dataset, visible_idcs)
                visibles.append(visible_selection)
            if invisible_idcs.shape[0] > 0:
                invisible_selection = Selection(dataset, invisible_idcs)
                invisibles.append(invisible_selection)

        return visibles, invisibles

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

    def make_vao(self):
        vao = QOpenGLVertexArrayObject()
        vao.create()
        return vao

    def init_vaos_and_buffers(self):
        for dataset, viewed_dataset in self._viewed_datasets.items():
            assert(dataset.n_dimensions() == 3) # for now?

            viewed_dataset['vao'] = self.make_vao()

            for dim in range(dataset.n_dimensions()):
                viewed_dataset['vbos'][dim] = self.make_vbo(dataset, dim)

    def enable_attributes(self, shader_program, gl):
        self.shader_program = shader_program

        for dataset, viewed_dataset in self._viewed_datasets.items():
            vao = self.make_vao()
            viewed_dataset['vao'] = vao

            vao.bind()

            for dim in range(dataset.n_dimensions()):
                vbo = self.make_vbo(dataset, dim)
                viewed_dataset['vbos'][dim] = vbo

            for attribute, vbo in zip(['position_x', 'position_y', 'color'], [viewed_dataset['vbos'][dim] for dim in range(dataset.n_dimensions())]):
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

    def draw(self, gl_widget):
        gl = gl_widget.gl
        shader_program = gl_widget.shader_program

        for dataset, viewed_dataset in self._viewed_datasets.items():
            if viewed_dataset['kind'] == 'regular':
                shader_program.setUniformValue('point_size', gl_widget.point_size)
            elif viewed_dataset['kind'] == 'representatives':
                shader_program.setUniformValue('point_size', 1.5*gl_widget.point_size)
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

    def destroy(self):
        if self.shader_program is not None:
            self.disable_attributes()
        for viewed_dataset in self._viewed_datasets.values():
            for vbo in viewed_dataset['vbos'].values():
                vbo.destroy()

    def __iter__(self):
        return iter(self._viewed_datasets.items())
