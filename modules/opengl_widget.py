from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class OpenGLWidget(QOpenGLWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.objects = set()

        self.model = QMatrix4x4()
        self.view = QMatrix4x4()
        self.projection = QMatrix4x4()

    def clear(self):
        for o in self.objects.copy():
            self.remove_object(o)

    def add_object(self, o):
        self.objects.add(o)

        self.makeCurrent()

        o.init_buffers()
        o.bind_to_shader(self.shader_program, self.gl)
        
        self.doneCurrent()
        
        self.update()

    def remove_object(self, o):
        self.objects.remove(o)

        self.makeCurrent()

        o.destroy_buffers()

        self.doneCurrent()
        
        self.update()

    def init_shaders(self, vertex_shader='shaders/vertex.glsl', fragment_shader='shaders/fragment.glsl'):
        self.shader_program = QOpenGLShaderProgram(self)

        # Read shader code from source
        with open(vertex_shader, 'r') as vs, open(fragment_shader) as fs:
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vs.read())
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fs.read())
        
        self.shader_program.link()

    def zoom(self, factor, pos):
        p_pixel = QVector3D(pos)
        
        # Manually transform from pixel coordinates to clip coordinates.
        p_pixel *= QVector3D(2 / self.width(), 2 / self.height(), 1)
        p_pixel -= QVector3D(1, 1, 0)
        p_clip = p_pixel * QVector3D(1, -1, 1)

        view_i, invertible = self.view.inverted()
        if not invertible:
            print('View matrix is not invertible.')
            return

        projection_i, invertible = self.projection.inverted()
        if not invertible:
            print('Projection matrix is not invertible.')
            return
        
        # Transform p from clip coordinates, through view coordinates, to world coordinates.
        p_world = view_i.map(projection_i.map(p_clip))

        # Change the view matrix s.t. it is zoomed in/out, but maps p to the same point.
        self.view.translate((1 - factor) * p_world)
        self.view.scale(factor)

        self.update()

    def set_pointsize(self, point_size):
        self.point_size = float(point_size)
        self.update()

    def mousePressEvent(self, mouse_press_event):
        p = QVector2D(mouse_press_event.pos())
        p *= QVector2D(1 / self.width(), 1 / self.height())

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
        gl = self.gl # Shorthand
        gl.initializeOpenGLFunctions()

        self.init_shaders()

        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE);
        gl.glEnable(gl.GL_MULTISAMPLE);
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);

    def paintGL(self):
        gl = self.gl # Shorthand
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.shader_program.bind()

        self.shader_program.setUniformValue('model', self.model)
        self.shader_program.setUniformValue('view', self.view)
        self.shader_program.setUniformValue('projection', self.projection)
        self.shader_program.setUniformValue('point_size', self.point_size)
        
        for o in self.objects:
            o.draw(gl)

        self.shader_program.release()

    def resizeGL(self, w, h):
        # Make new projection matrix to retain aspect ratio
        self.projection.setToIdentity();
        if w > h:
            self.projection.scale(h / w, 1.0);
        else:
            self.projection.scale(1.0, w / h);
        
        # Define the new viewport (is this even necessary?)
        self.gl.glViewport(0, 0, w, h)

    def setClearColor(self, c):
        self.gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        self.gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())
