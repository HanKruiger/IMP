import numpy as np
from model import *
from widgets import *
from renderers import DatasetViewRenderer

class GLWidget(QOpenGLWidget):

    def __init__(self, imp_window):
        super().__init__()
        self.imp_window = imp_window
        self.setMouseTracking(True)

        self.dataset_view_renderer = DatasetViewRenderer(self)

    def previous_view(self):
        self.dataset_view_renderer.previous()

    def mouseMoveEvent(self, e):
        mouse = QVector2D(e.pos())
        if e.buttons() == Qt.LeftButton:
            self.dataset_view_renderer.translate(mouse - self.mouse_when_clicked)
            self.mouse_when_clicked = mouse
            self.update()

    def mousePressEvent(self, e):
        self.mouse_when_clicked = QVector2D(e.pos())

    def wheelEvent(self, wheel_event):
        if wheel_event.pixelDelta().y() == 0:
            wheel_event.ignore()
            return
        wheel_event.accept()

        factor = 1.01 ** wheel_event.pixelDelta().y()
        if QGuiApplication.keyboardModifiers() == Qt.ControlModifier:
            if factor > 1:
                self.imp_window.datasets_widget.hierarchical_zoom()
        elif QGuiApplication.keyboardModifiers() == Qt.ShiftModifier:
            if factor > 1:
                self.dataset_view_renderer.next()
            else:
                self.dataset_view_renderer.previous()
        else:
            self.dataset_view_renderer.zoom(factor, wheel_event.pos())
            self.update()

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def initializeGL(self):
        self.gl = self.context().versionFunctions()
        gl = self.gl  # Shorthand
        gl.initializeOpenGLFunctions()

        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.dataset_view_renderer.init_gl(gl)

    def paintGL(self):
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)
        self.dataset_view_renderer.draw()

    def resizeGL(self, w, h):
        self.dataset_view_renderer.resize(w, h)
        self.gl.glViewport(0, 0, w, h)

    def setClearColor(self, c):
        self.gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.view_transitionF())

    def setColor(self, c):
        self.gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.view_transitionF())
