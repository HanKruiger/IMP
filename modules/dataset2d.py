from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

class Dataset2D:

    def __init__(self, Y):
        self.Y = Y
        self.colors = np.random.random((Y.shape[0], 4)) # for now

    def init_shaders(self, parent, vertex_shader='shaders/vertex.glsl', fragment_shader='shaders/fragment.glsl'):
        self.program = QOpenGLShaderProgram(parent)

        # Read shader code from source
        with open(vertex_shader, 'r') as vs, open(fragment_shader) as fs:
            self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vs.read())
            self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fs.read())
        
        self.program.link()

    def init_buffers(self, gl):
        # Make (temporary) 32-bit duplicates
        Y_32 = np.array(self.Y, dtype=np.float32)
        colors_32 = np.array(self.colors, dtype=np.float32)

        # Make a VAO that contains all arrays for our triangle.
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        # Make the VBO that contains the vertex coordinates
        # Also, fill it with the position data.
        self.position_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.position_vbo.create()
        self.position_vbo.bind()
        self.position_vbo.setUsagePattern(QOpenGLBuffer.DynamicDraw)
        self.position_vbo.allocate(Y_32.data.nbytes)
        self.position_vbo.write(0, Y_32, Y_32.data.nbytes)
        self.position_vbo.release()

        # Make the VBO that contains the vertex colours
        # Also, fill it with the colour data.
        self.color_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.color_vbo.create()
        self.color_vbo.bind()
        self.color_vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self.color_vbo.allocate(colors_32.data, colors_32.data.nbytes)
        self.color_vbo.release()

        # Get the 'internal addresses' of the required attributes in the shader
        self.program.bind()
        self.position_attr = self.program.attributeLocation('position')
        self.color_attr = self.program.attributeLocation('color')
        
        # Explain the format of the 'position' attribute buffer to the shader.
        self.program.enableAttributeArray(self.position_attr)
        self.position_vbo.bind()
        self.program.setAttributeBuffer(
            self.position_attr,    # Attribute location
            gl.GL_FLOAT,       # Data type of elements
            0,                      # Offset
            2,                      # Number of components per vertex (here: x, y)
            0                       # Stride
        )
        self.position_vbo.release()

        # Explain the format of the 'colour' attribute buffer to the shader.
        self.program.enableAttributeArray(self.color_attr)
        self.color_vbo.bind()
        self.program.setAttributeBuffer(
            self.color_attr,  # Attribute location
            gl.GL_FLOAT,   # Data type of elements
            0,                  # Offset
            4,                  # Number of components per vertex (here: RGBA)
            0                   # Stride
        )
        self.color_vbo.release()

        # NOTE: The VAO remembers:
        #   * which buffers are enabled/disabled
        #   * which attribute arrays are enabled/disabled for which buffers

        self.program.release()
        self.vao.release()

        self.program.bind()
        gl.glUniform1f(self.program.uniformLocation('point_size'), 8)
        self.program.release()

    def update_Y(self, Y):
        self.Y = Y
        Y_32 = np.array(self.Y, dtype=np.float32)

        # Update the buffer
        self.position_vbo.bind()
        if self.position_vbo.size() != Y_32.data.nbytes:
            print('Reallocating.')
            self.position_vbo.allocate(Y_32.data, Y_32.data.nbytes)
        else:
            # No need to reallocate. Just overwrite
            print('Overwriting.')
            self.position_vbo.write(0, Y_32, Y_32.data.nbytes)
        self.position_vbo.release()
            

    def draw(self, gl):
        self.vao.bind()
        self.program.bind()
        gl.glDrawArrays(gl.GL_POINTS, 0, self.Y.shape[0])
        self.program.release()
        self.vao.release()