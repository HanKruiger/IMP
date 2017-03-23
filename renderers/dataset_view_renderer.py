from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *

class DatasetViewRenderer(QObject):

    animation_done = pyqtSignal()

    def __init__(self, gl_widget):
        super().__init__()
        self.gl_widget = gl_widget
        self.dataset_views = []
        self.current_view = None

        # Transforms points from world space to view space
        self.view = QMatrix4x4()

        # Transforms points from view space to clip space
        self.projection = QMatrix4x4()

        # Transforms points from screen space to clip space
        # (inverse viewport transform)
        self.pixel = QMatrix4x4()

        self.fadein_interpolation = 0.0
        self.fadein_animation_timer = QTimer()
        self.fadein_animation_timer.timeout.connect(self.fadein_animation)


    def vis_params(self):
        return self.gl_widget.imp_window.vis_params()

    def init_gl(self, gl):
        self.gl = gl
        self.init_shaders()

    def init_shaders(self):
        vertex_shader='shaders/points/vertex.glsl'
        fragment_shader='shaders/points/fragment.glsl'

        self.shader_program = QOpenGLShaderProgram(self.gl_widget)

        # Read shader code from source
        with open(vertex_shader, 'r') as vs, open(fragment_shader) as fs:
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vs.read())
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fs.read())

        self.shader_program.link()

    def get_latest(self):
        try:
            return self.dataset_views[-1]
        except IndexError:
            return None

    def previous(self):
        try:
            self.show_dataset_view(self.current_view.previous())
        except AttributeError:
            pass

    def next(self):
        try:
            self.show_dataset_view(self.current_view.next())
        except AttributeError:
            pass

    def add_to_schedule(self, dataset_view):
        self.dataset_views.append(dataset_view)

    def remove_dataset(self, dataset):
        if dataset in self.current_view.datasets():
            self.clear_dataset_view(self.current_view)

    def show_dataset(self, dataset, representatives=None, fit_to_view=False):
        dataset_view = DatasetView(previous=self.get_latest())
        dataset_view.add_dataset(dataset, 'regular')
        if representatives is not None:
            dataset_view.add_dataset(representatives, 'representatives')

        self.show_dataset_view(dataset_view, fit_to_view=fit_to_view)
        if representatives is not None:
            self.fadein_interpolation = 0.0
            self.fadein_animation_timer.start()
        else:
            self.fadein_interpolation = 1.0


    def show_dataset_view(self, dataset_view, fit_to_view=False):
        self.dataset_views.append(dataset_view)
        self.set_current_view(dataset_view, fit_to_view=fit_to_view)

    def set_dataset_view(self, fit_to_view=False):
        dataset_view = self.dataset_views[-1]

    def set_current_view(self, dataset_view, fit_to_view=False):
        if self.current_view is not None:
            self.clear_dataset_view(self.current_view)

        self.gl_widget.makeCurrent()
        dataset_view.init_vaos_and_buffers()
        dataset_view.enable_attributes(self.shader_program, self.gl)
        self.gl_widget.doneCurrent()

        if fit_to_view:
            x_min, x_max, y_min, y_max = dataset_view.get_bounds()

            # Compute the center of the all-enclosing square, and the distance from center to its sides.
            center = QVector2D((x_min + x_max) / 2, (y_min + y_max) / 2)
            dist_to_sides = 0.5 * max(x_max - x_min, y_max - y_min)

            # To account for points that are almost exactly at the border, we need a small
            # tolerance value. (Can also be seen as (very small) padding.)
            tolerance = .2
            dist_to_sides *= 1 + tolerance

            self.view.setToIdentity()
            self.view.scale(1 / dist_to_sides)
            self.view.translate(-center.x(), -center.y())

        self.gl_widget.update()

        self.current_view = dataset_view

    def clear_dataset_view(self, dataset_view):
        self.gl_widget.makeCurrent()
        dataset_view.disable()
        self.gl_widget.doneCurrent()
        self.gl_widget.update()

    def draw(self):
        self.shader_program.bind()

        self.shader_program.setUniformValue('projection', self.projection)
        self.shader_program.setUniformValue('opacity', self.vis_params().get('opacity'))
        self.shader_program.setUniformValue('point_size', self.vis_params().get('point_size'))
        self.shader_program.setUniformValue('view', self.view)
        self.shader_program.setUniformValue('fadein_interpolation', self.fadein_interpolation)

        if len(self.dataset_views) > 0 and self.dataset_views[-1].is_active():
            self.dataset_views[-1].draw(self.gl, self.shader_program)

        self.shader_program.release()

    @pyqtSlot()
    def zoom(self, factor, pos):
        p_world = self.pixel_to_world(pos, d=3)
        self.view.translate((1 - factor) * p_world)
        self.view.scale(factor)

    def translate(self, movement):
        self.view.translate(self.pixel_to_world(movement, d=3, w=0))

    def fadein_animation(self):
        self.fadein_interpolation += 0.01
        if self.fadein_interpolation >= 1:
            self.fadein_interpolation = 1
            self.fadein_animation_timer.stop()
        self.gl_widget.update()

    def pixel_to_world(self, p_in, d=2, w=1):
        try:
            scalar = float(p_in)
            p_pixel = QVector4D(scalar, 0, 0, w)
        except TypeError:
            p_pixel = QVector4D(p_in)
            if not isinstance(p_in, QVector4D):
                p_pixel.setW(w)

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

    def resize(self, w, h):
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

    def filter_unseen_points(self):
        return self.dataset_views[-1].filter_unseen_points(self.projection * self.view)

    def current_union(self):
        return self.dataset_views[-1].union()
