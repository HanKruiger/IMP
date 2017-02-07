from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

class Lense:

    def __init__(self, gl_widget):
        self.gl_widget = gl_widget
        self.radius = 150.0

    def init_gl(self):
        self.gl = self.gl_widget.gl
        self.init_vao()
        self.init_shader()
        self.init_buffer()
        self.enable_attributes()

    def init_vao(self):
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()

    def init_shader(self, vertex_shader='shaders/lense/vertex.glsl', fragment_shader='shaders/lense/fragment.glsl'):
        self.shader_program = QOpenGLShaderProgram(self.gl_widget)

        # Read shader code from source
        with open(vertex_shader, 'r') as vs, open(fragment_shader) as fs:
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vs.read())
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fs.read())

        self.shader_program.link()

    def init_buffer(self):
        X_32 = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)

        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self.vbo.allocate(X_32.data, X_32.data.nbytes)
        self.vbo.release()

    def enable_attributes(self):
        self.vao.bind()

        self.shader_program.bind()
        pos_loc = self.shader_program.attributeLocation('a_position')
        self.shader_program.enableAttributeArray(pos_loc)
        self.vbo.bind()
        self.shader_program.setAttributeBuffer(
            pos_loc,    # Attribute location
            self.gl.GL_FLOAT,       # Data type of elements
            0,                      # Offset
            2,                      # Number of components per vertex
            0                       # Stride
        )
        self.vbo.release()
        self.shader_program.release()

        self.vao.release()

    def draw(self):
        p_world = self.world_coordinates()
        print('x:{} y:{} r:{}'.format(p_world.x(), p_world.y(), self.world_radius()))
        self.shader_program.bind()
        self.shader_program.setUniformValue('u_radius', self.radius)
        self.shader_program.setUniformValue('u_center', self.gl_widget.mouse)
        self.vao.bind()
        self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0, 4)
        self.vao.release()
        self.shader_program.release()

    def world_coordinates(self):
        return self.gl_widget.pixel_to_world(self.gl_widget.mouse)

    def world_radius(self):
        return abs(self.gl_widget.pixel_to_world(self.radius, d=1))
