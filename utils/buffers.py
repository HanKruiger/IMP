from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

def make_vbo(data, normalize=False):
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

def update_vbo(vbo, data, normalize=False):
    data = np.atleast_2d(data.copy())

    if normalize:
        for dim in range(data.shape[1]):
            data[:, dim] -= data[:, dim].min()
            data[:, dim] /= data[:, dim].max()

    vbo.bind()
    vbo.allocate(data.data, data.data.nbytes)
    vbo.release()

    return vbo

def make_vao():
    vao = QOpenGLVertexArrayObject()
    vao.create()
    return vao