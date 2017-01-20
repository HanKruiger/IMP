from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

class Dataset2D:

    def __init__(self, Y):
        self.Y = Y
        rgb = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.5],
            [0, 0, 1, 0.5]
        ])
        self.colors = rgb[np.random.choice(3, (Y.shape[0], 1))] # for now
        self.init_buffers()

    def init_buffers(self):
        # Make (temporary) 32-bit duplicates
        Y_32 = np.array(self.Y, dtype=np.float32)
        colors_32 = np.array(self.colors, dtype=np.float32)

        # Make a VAO. It will remember the enabled attributes
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()

        # Make the VBO that contains the vertex coordinates
        # Also, fill it with the position data.
        self.position_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.position_vbo.create()
        self.position_vbo.bind()
        self.position_vbo.setUsagePattern(QOpenGLBuffer.DynamicDraw)
        self.position_vbo.allocate(Y_32.data, Y_32.data.nbytes)
        self.position_vbo.release()

        # Make the VBO that contains the vertex colours
        # Also, fill it with the colour data.
        self.color_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.color_vbo.create()
        self.color_vbo.bind()
        self.color_vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self.color_vbo.allocate(colors_32.data, colors_32.data.nbytes)
        self.color_vbo.release()

    def bind_to_shader(self, shader_program, gl):
        # Bind the VAO. It will remember the enabled attributes
        self.vao.bind()
        
        # Get the 'internal addresses' of the required attributes in the shader
        shader_program.bind()
        self.position_attr = shader_program.attributeLocation('position')
        self.color_attr = shader_program.attributeLocation('color')
        
        # Explain the format of the 'position' attribute buffer to the shader.
        shader_program.enableAttributeArray(self.position_attr)
        self.position_vbo.bind()
        shader_program.setAttributeBuffer(
            self.position_attr,    # Attribute location
            gl.GL_FLOAT,       # Data type of elements
            0,                      # Offset
            2,                      # Number of components per vertex (here: x, y)
            0                       # Stride
        )
        self.position_vbo.release()

        # Explain the format of the 'colour' attribute buffer to the shader.
        shader_program.enableAttributeArray(self.color_attr)
        self.color_vbo.bind()
        shader_program.setAttributeBuffer(
            self.color_attr,  # Attribute location
            gl.GL_FLOAT,   # Data type of elements
            0,                  # Offset
            4,                  # Number of components per vertex (here: RGBA)
            0                   # Stride
        )
        self.color_vbo.release()
        shader_program.release()
        
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
        gl.glDrawArrays(gl.GL_POINTS, 0, self.Y.shape[0])
        self.vao.release()
