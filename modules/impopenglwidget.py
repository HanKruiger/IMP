from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ImpOpenGLWidget(QOpenGLWidget):

    def __init__(self, parent=None):
        super(ImpOpenGLWidget, self).__init__(parent)
        self.objects = []
        self.camera = QMatrix4x4()

    def add_object(self, o):
        self.objects.append(o)
        o.init_shaders(self)
        o.init_buffers(self.gl)

    def mousePressEvent(self, event):
        print('{}, {}'.format(event.x() / self.width(), event.y() / self.height()))

    def wheelEvent(self, event):
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 1 / factor

        # Manually transform from pixel coords to -1 - +1 coordinates.
        p = QVector3D(event.x() / self.width(), event.y() / self.height(), 0)
        p *= 2
        p -= QVector3D(1, 1, 0)
        p.setY(-p.y())

        # Perform the inverse of the camera projection
        camera_i, invertible = self.camera.inverted()
        if not invertible:
            print('Camera matrix is not invertible.')
            return

        # TODO: Zoom s.t. mouse stays on same point!
        p = camera_i.mapVector(p)
        self.camera.translate(p)
        self.camera.scale(factor)
        self.camera.translate(-p)

        self.update()

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def initializeGL(self):
        self.gl = self.context().versionFunctions()
        gl = self.gl # Shorthand
        gl.initializeOpenGLFunctions()

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE);
        gl.glEnable(gl.GL_MULTISAMPLE);
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE);

    def paintGL(self):
        gl = self.gl # Shorthand
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for o in self.objects:
            o.draw(gl)

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        self.gl.glViewport((width - side) // 2, (height - side) // 2, side, side)
        print('Resized to {0} by {1}.'.format(width, height))

    def setClearColor(self, c):
        self.gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        self.gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())
