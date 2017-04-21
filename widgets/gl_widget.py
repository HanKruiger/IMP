import numpy as np
from model import *
from widgets import *
from renderers import DatasetViewRenderer

class GLWidget(QOpenGLWidget):

    def __init__(self, imp_window):
        super().__init__()
        self.imp_window = imp_window
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.dataset_view_renderer = DatasetViewRenderer(self)

    def previous_view(self):
        self.dataset_view_renderer.previous()

    def mouseMoveEvent(self, e):
        self.mouse_pos = QVector2D(e.pos())
        if e.buttons() == Qt.LeftButton:
            self.dataset_view_renderer.translate(self.mouse_pos - self.mouse_when_clicked)
            self.mouse_when_clicked = self.mouse_pos
            self.update()

    def mousePressEvent(self, e):
        self.mouse_when_clicked = QVector2D(e.pos())

    def hierarchical_zoom(self):
        world_pos = self.dataset_view_renderer.pixel_to_world(self.mouse_pos)
        world_pos = np.array([world_pos.x(), world_pos.y()])
        self.imp_window.datasets_widget.hierarchical_zoom(world_pos)

    def wheelEvent(self, wheel_event):
        if wheel_event.pixelDelta().y() == 0:
            wheel_event.ignore()
            return
        wheel_event.accept()

        factor = 1.01 ** wheel_event.pixelDelta().y()
        # self.dataset_view_renderer.zoom(factor, wheel_event.pos())
        if factor > 1:
            self.hierarchical_zoom()

        self.update()

    def keyPressEvent(self, key_event):
        if key_event.key() == Qt.Key_Z:
            key_event.accept()
            self.hierarchical_zoom()
        elif key_event.key() == Qt.Key_Comma:
            self.dataset_view_renderer.previous()
        elif key_event.key() == Qt.Key_Period:
            self.dataset_view_renderer.next()

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
