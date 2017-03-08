from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np


class DatasetsView:

    def __init__(self):
        self._viewed_datasets = dict()

    def add_dataset(self, dataset, kind):
        self._viewed_datasets[dataset] = {
            'kind': kind,
            'vao': None,
            'vbos': dict()
        }

    def datasets(self):
        return [dataset for dataset in self._viewed_datasets.keys()]

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
            assert(dataset.m == 3) # for now?

            viewed_dataset['vao'] = self.make_vao()

            for dim in range(dataset.m):
                viewed_dataset['vbos'][dim] = self.make_vbo(dataset, dim)

    def enable_attributes(self, shader_program, gl):
        for dataset, viewed_dataset in self._viewed_datasets.items():
            vao = self.make_vao()
            viewed_dataset['vao'] = vao

            vao.bind()

            for dim in range(dataset.m):
                vbo = self.make_vbo(dataset, dim)
                viewed_dataset['vbos'][dim] = vbo

            for attribute, vbo in zip(['position_x', 'position_y', 'color'], [viewed_dataset['vbos'][dim] for dim in range(dataset.m)]):
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
            gl.glDrawArrays(gl.GL_POINTS, 0, dataset.N)
            vao.release()

    def disable_attributes(self, shader_program):
        for dataset, viewed_dataset in self._viewed_datasets.items():
            vao = viewed_dataset['vao']

            vao.bind()

            for attribute in ['position_x', 'position_y', 'color']:
                shader_program.disableAttributeArray(attribute)

            vao.release()

    def destroy(self):
        for viewed_dataset in self._viewed_datasets.values():
            for vbo in viewed_dataset['vbos'].values():
                vbo.destroy()

    def __iter__(self):
        return iter(self._viewed_datasets.items())
