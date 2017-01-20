from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ImpOpenGLWidget(QOpenGLWidget):

    def __init__(self, parent=None):
        super(ImpOpenGLWidget, self).__init__(parent)
        self.objects = []
        self.model = QMatrix4x4()
        self.model.scale(2, 2);
        self.model.translate(-0.5, -0.5);
        self.view = QMatrix4x4()
        self.projection = QMatrix4x4()

    def add_object(self, o):
        self.objects.append(o)
        o.bind_to_shader(self.shader_program, self.gl)

    def init_shaders(self, vertex_shader='shaders/vertex.glsl', fragment_shader='shaders/fragment.glsl'):
        self.shader_program = QOpenGLShaderProgram(self)

        # Read shader code from source
        with open(vertex_shader, 'r') as vs, open(fragment_shader) as fs:
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vs.read())
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fs.read())
        
        self.shader_program.link()

    def mousePressEvent(self, event):
        print('{}, {}'.format(event.x() / self.width(), event.y() / self.height()))

    def wheelEvent(self, wheel_event):
        factor = 1.1
        if wheel_event.angleDelta().y() < 0:
            factor = 1 / factor

        # Manually transform from pixel coordinates to clip coordinates.
        p = QVector3D(wheel_event.pos())
        p *= QVector3D(2 / self.width(), 2 / self.height(), 1)
        p -= QVector3D(1, 1, 0)
        p_clip = p * QVector3D(1, -1, 1)

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

        # Change the view matrix s.t. it is zoomed in, but maps p to the same point.
        self.view.translate((1 - factor) * p_world)
        self.view.scale(factor)

        self.update()

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def initializeGL(self):
        self.gl = self.context().versionFunctions()
        gl = self.gl # Shorthand
        gl.initializeOpenGLFunctions()

        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE);
        gl.glEnable(gl.GL_MULTISAMPLE);
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);

        self.init_shaders()

    def paintGL(self):
        gl = self.gl # Shorthand
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.shader_program.bind()
        self.shader_program.setUniformValue('model', self.model)
        self.shader_program.setUniformValue('view', self.view)
        self.shader_program.setUniformValue('projection', self.projection)
        self.shader_program.setUniformValue('point_size', 8.0)
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
