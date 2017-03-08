from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
from widgets.gl_entities.lense import Lense
from operators.selectors import LenseSelector
from model.dataset import Embedding

class OpenGLWidget(QOpenGLWidget):

    def __init__(self, imp_app):
        super().__init__()
        self.imp_app = imp_app
        self.setMouseTracking(True)

        self.view = QMatrix4x4()
        self.view.scale(0.25, 0.25)
        self.projection = QMatrix4x4()
        self.pixel = QMatrix4x4()

        self.mouse = QVector2D(0, 0)

        self.lense = Lense(self)

        self.attributes = dict()
        self.datasets_view = None

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

    def set_datasets_view(self, datasets_view):
        if self.datasets_view is not None:
            self.disable_attributes()
            self.datasets_view.destroy()

        self.datasets_view = datasets_view

        offset = 0
        for dataset, viewed_dataset in datasets_view:
            kind = viewed_dataset['kind']
            vbos = viewed_dataset['vbos']
            self.set_attribute(vbos[0], 0, dataset.N, 1, 'position_x', offset)
            self.set_attribute(vbos[1], 1, dataset.N, 1, 'position_y', offset)
            self.set_attribute(vbos[2], 2, dataset.N, 1, 'color', offset)
            offset += dataset.N

    def set_attribute(self, vbo, dim, N, m, attribute, offset):
        self.makeCurrent()

        # Bind the VAO. It will remember the enabled attributes
        self.vao.bind()

        # Get the 'internal addresses' of the required attributes in the shader
        self.shader_program.bind()
        attrib_loc = self.shader_program.attributeLocation(attribute)

        # Explain the format of the attribute buffer to the shader.
        self.shader_program.enableAttributeArray(attrib_loc)
        vbo.bind()
        self.shader_program.setAttributeBuffer(
            attrib_loc,    # Attribute location
            self.gl.GL_FLOAT,       # Data type of elements
            offset,                      # Offset
            m,                      # Number of components per vertex
            0                       # Stride
        )
        vbo.release()

        self.shader_program.release()

        self.vao.release()
        self.doneCurrent()
        self.update()

    def disable_attributes(self):
        self.makeCurrent()
        self.vao.bind()
        self.shader_program.bind()
        self.shader_program.disableAttributeArray('position_x')
        self.shader_program.disableAttributeArray('position_y')
        self.shader_program.disableAttributeArray('color')
        self.shader_program.release()
        self.vao.release()
        self.doneCurrent()
        self.update()

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
       
        center = self.lense.world_coordinates()
        radius = self.lense.world_radius()
        dataset = self.attributes['position_x']['dataset']
        self.imp_app.datasets_widget.hierarchical_zoom(dataset, self.attributes['position_x']['dim'], self.attributes['position_y']['dim'], center, radius)
        

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
        offset = 0
        if self.datasets_view is not None:
            for dataset, viewed_dataset in self.datasets_view:
                kind = viewed_dataset['kind']
                print(kind)
                gl.glDrawArrays(gl.GL_POINTS, offset, dataset.N)
                offset += dataset.N
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
