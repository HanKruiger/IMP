from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
from modules.dataset import Embedding
from modules.lense import Lense
from modules.selectors import LenseSelector

class OpenGLWidget(QOpenGLWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(True)

        self.view = QMatrix4x4()
        self.view.scale(0.25, 0.25)
        self.projection = QMatrix4x4()
        self.pixel = QMatrix4x4()

        self.mouse = QVector2D(0, 0)

        self.lense = Lense(self)

        self.attributes = dict()

        self.N = 0

    def init_shaders(self, vertex_shader='shaders/points/vertex.glsl', fragment_shader='shaders/points/fragment.glsl'):
        self.shader_program = QOpenGLShaderProgram(self)

        # Read shader code from source
        with open(vertex_shader, 'r') as vs, open(fragment_shader) as fs:
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vs.read())
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fs.read())

        self.shader_program.link()
        

    def init_vao(self):
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()

    def clear(self):
        pass

    def set_attribute(self, dataset, dim, N, m, attribute):
        # Remove this attribute if it is already set.
        if attribute in self.attributes:
            self.disable_attribute(attribute)


        # Loop over attributes and remove the ones incompatible with N
        for attr, descr in self.attributes.copy().items():
            if descr['size'] != N:
                self.disable_attribute(attr)

        self.N = N
        self.attributes[attribute] = {'dataset': dataset, 'dim': dim, 'size': N}

        if attribute == 'color':
            normalize = True
        else:
            normalize = False

        self.makeCurrent()

        # Get the VBO from the dataset (will be generated if it doesn't exist)
        if not normalize:
            vbo = dataset.vbo(dim)
        else:
            vbo = dataset.normalized_vbo(dim)

        # Bind the VAO. It will remember the enabled attributes
        self.vao.bind()

        # Get the 'internal addresses' of the required attributes in the shader
        self.shader_program.bind()
        attrib_loc = self.shader_program.attributeLocation(attribute)

        # Explain the format of the 'position' attribute buffer to the shader.
        self.shader_program.enableAttributeArray(attrib_loc)
        vbo.bind()
        self.shader_program.setAttributeBuffer(
            attrib_loc,    # Attribute location
            self.gl.GL_FLOAT,       # Data type of elements
            0,                      # Offset
            m,                      # Number of components per vertex
            0                       # Stride
        )
        vbo.release()

        self.shader_program.release()

        self.vao.release()
        self.doneCurrent()
        self.update()

    def disable_attribute(self, attribute):
        try:
            del self.attributes[attribute]
            if not self.attributes:
                self.N = 0

            self.makeCurrent()
            self.vao.bind()
            self.shader_program.bind()
            self.shader_program.disableAttributeArray(attribute)
            self.shader_program.release()
            self.vao.release()
            self.doneCurrent()
            self.update()

        except KeyError:
            print('Tried to disable attrubute that was not enabled.')

    def zoom(self, factor, pos):
        p_world = self.pixel_to_world(pos, d=3)

        # Change the view matrix s.t. it is zoomed in/out, but maps p to the same point.
        self.view.translate((1 - factor) * p_world)
        self.view.scale(factor)

        self.update()

    def pixel_to_world(self, p_in, d=2):
        try:
            scalar = float(p_in)
            p_pixel = QVector4D(scalar, 0, 0, 0) # w = 0, since we want to transform a distance (no translations!).
        except TypeError:
            p_pixel = QVector4D(p_in)
            if not isinstance(p_in, QVector4D):
                p_pixel.setW(1) # w = 1, since we assume a vector transformation (yes translations!).

        pixel_i, invertible = self.pixel.inverted()
        if not invertible:
            print('Pixel matrix is not invertible.')
            return

        view_i, invertible = self.view.inverted()
        if not invertible:
            print('View matrix is not invertible.')
            return

        projection_i, invertible = self.projection.inverted()
        if not invertible:
            print('Projection matrix is not invertible.')
            return

        p_world = view_i.map(projection_i.map(pixel_i.map(p_pixel)))
        
        if d == 4:
            return p_world
        elif d == 3:
            return p_world.toVector3D()
        elif d == 2:
            return p_world.toVector2D()
        elif d == 1:
            return p_world.length()
            

    def set_pointsize(self, point_size):
        self.point_size = float(point_size)**0.5
        self.update()

    # Receives value in 0-255 range
    def set_opacity(self, opacity):
        self.opacity = opacity / 255
        self.update()

    def mouseMoveEvent(self, e):
        self.mouse = QVector2D(e.pos())
        self.update()

    def mousePressEvent(self, e):
        self.mouse = QVector2D(e.pos())
       
        p_world = self.lense.world_coordinates()
        r_world = self.lense.world_radius()

        dataset = self.attributes['position_x']['dataset']
        parent = dataset

        # We want to select in 'embedding', but make the selection results in the nD parent.
        while type(parent) == Embedding:
            parent = parent.parent()

        selector = LenseSelector()
        selector.set_input({
            'embedding': dataset,
            'parent': parent
        })
        selector.set_parameters({
            'lense': self.lense,
            'x_dim': self.attributes['position_x']['dim'],
            'y_dim': self.attributes['position_y']['dim']
        })

        parent.perform_operation(selector)

        self.update()

    def wheelEvent(self, wheel_event):
        if wheel_event.pixelDelta().y() == 0:
            wheel_event.ignore()
            return
        wheel_event.accept()
        factor = 1.01 ** wheel_event.pixelDelta().y()
        self.zoom(factor, wheel_event.pos())

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def initializeGL(self):
        self.gl = self.context().versionFunctions()
        gl = self.gl  # Shorthand
        gl.initializeOpenGLFunctions()

        self.init_shaders()
        self.init_vao()

        self.lense.init_gl()

        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def paintGL(self):
        gl = self.gl  # Shorthand
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.shader_program.bind()

        self.shader_program.setUniformValue('view', self.view)
        self.shader_program.setUniformValue('projection', self.projection)
        self.shader_program.setUniformValue('point_size', self.point_size)
        self.shader_program.setUniformValue('opacity', self.opacity)

        self.vao.bind()
        gl.glDrawArrays(gl.GL_POINTS, 0, self.N)
        self.vao.release()

        self.shader_program.release()

        self.lense.draw()


    def resizeGL(self, w, h):
        # Make new projection matrix to retain aspect ratio
        self.projection.setToIdentity()
        if w > h:
            self.projection.scale(h / w, 1.0)
        else:
            self.projection.scale(1.0, w / h)

        self.pixel.setToIdentity()
        self.pixel.scale(0.5 * w, 0.5 * h)
        self.pixel.translate(1, 1)
        self.pixel.scale(1, -1)

        # Define the new viewport (is this even necessary?)
        self.gl.glViewport(0, 0, w, h)

    def setClearColor(self, c):
        self.gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        self.gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())
