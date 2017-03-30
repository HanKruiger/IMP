from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *


class DatasetViewRenderer(QObject):

    animation_done = pyqtSignal()

    def __init__(self, gl_widget):
        super().__init__()
        self.gl_widget = gl_widget

        # Transforms points from world space to view space
        # self.view_matrix() (this is per datasetview object!)

        # Transforms points from view space to clip space
        self.projection = QMatrix4x4()

        # Transforms points from screen space to clip space
        # (inverse viewport transform)
        self.pixel = QMatrix4x4()

        self.interpolation_time = 200

        self.animation_time = 0.0
        self.animation_timer = Timer(n_steps=60)
        self.animation_timer.tick.connect(self.set_animation_time)

    def vis_params(self):
        return self.gl_widget.imp_window.vis_params()

    def init_gl(self, gl):
        self.gl = gl
        self.init_shaders()

    def init_shaders(self):
        vertex_shader = 'shaders/points/vertex.glsl'
        fragment_shader = 'shaders/points/fragment.glsl'

        self.shader_program = QOpenGLShaderProgram(self.gl_widget)

        # Read shader code from source
        with open(vertex_shader, 'r') as vs, open(fragment_shader) as fs:
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vs.read())
            self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fs.read())

        self.shader_program.link()

    def current_view(self):
        try:
            return self._current_view
        except AttributeError:
            return None

    def previous(self):
        try:
            previous_view = self.current_view().previous()
            def callback():
                self.show_dataset_view(previous_view, forward=False)
                self.animation_timer.stopped.disconnect(callback)
            self.animation_timer.stopped.connect(callback)
            self.animation_timer.start(self.interpolation_time, forward=False)
        except AttributeError:
            pass

    def next(self):
        try:
            next_view = self.current_view().next()
            self.show_dataset_view(next_view, forward=True)
            self.animation_timer.start(self.interpolation_time, forward=True)
        except AttributeError:
            pass

    def remove_dataset(self, dataset):
        if dataset in self.current_view().datasets():
            self.disable_dataset_view(self.current_view())

    def show_dataset(self, dataset, representatives=None, fit_to_view=False):
        dataset_view = DatasetView(previous=self.current_view())
        dataset_view.set_new_regular(dataset)
        if representatives is not None:
            dataset_view.set_new_representative(representatives)

        self.show_dataset_view(dataset_view, fit_to_view=fit_to_view, forward=False)


    def interpolate_to_dataset(self, dataset, representatives, forward=True):
        dataset_view = DatasetView(previous=self.current_view())

        dataset_view.set_old_regular(self.current_view().new_regular())
        dataset_view.set_new_regular(dataset)
        dataset_view.set_old_representative(self.current_view().new_representative())
        dataset_view.set_new_representative(representatives)

        self.show_dataset_view(dataset_view, forward=forward, fit_to_view=True)
        self.animation_time = 0.0
        self.animation_timer.start(self.interpolation_time, forward=forward)

    def show_dataset_view(self, dataset_view, fit_to_view=False, forward=True):
        try:
            self.disable_dataset_view(self.current_view())
        except AttributeError:
            pass

        self.gl_widget.makeCurrent()
        dataset_view.enable(self.shader_program, self.gl)
        self.gl_widget.doneCurrent()

        if fit_to_view:
            dataset_view.fit_to_view()

        self._current_view = dataset_view

        if forward:
            self.animation_time = 0.0
        else:
            self.animation_time = 1.0

        self.gl_widget.update()

    def disable_dataset_view(self, dataset_view):
        self.gl_widget.makeCurrent()
        dataset_view.disable()
        self.gl_widget.doneCurrent()

        self.gl_widget.update()

    def draw(self):
        self.shader_program.bind()

        self.shader_program.setUniformValue('u_projection', self.projection)
        self.shader_program.setUniformValue('u_opacity_regular', self.vis_params().get('opacity_regular'))
        self.shader_program.setUniformValue('u_opacity_representatives', self.vis_params().get('opacity_representatives'))
        self.shader_program.setUniformValue('u_point_size', self.vis_params().get('point_size'))
        self.shader_program.setUniformValue('u_time', self.animation_time)

        try:
            self.current_view().draw(self.gl)
        except AttributeError:
            pass

        self.shader_program.release()

    @pyqtSlot()
    def zoom(self, factor, pos):
        p_world = self.pixel_to_world(pos, d=3)
        self.view_matrix('new').translate((1 - factor) * p_world)
        self.view_matrix('new').scale(factor)

    def translate(self, movement):
        self.view_matrix('new').translate(self.pixel_to_world(movement, d=3, w=0))

    def view_matrix(self, old_or_new='new'):
        return self.current_view().view_matrix(old_or_new)

    def set_animation_time(self, time):
        self.animation_time = time
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

        view_i, invertible = self.view_matrix('new').inverted()
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

    def filter_unseen_points(self, view=None):
        if view is None:
            view = self.view_matrix('new')

        projection_view = self.projection * view
        projection_view = np.array(projection_view.data()).reshape((4, 4))

        # Delete z-entries in transformation matrix (it's 2D, not 3D).
        projection_view = np.delete(projection_view, 2, axis=0)
        projection_view = np.delete(projection_view, 2, axis=1)

        # Build union of all datasets in the current DatasetView
        union = self.current_union()

        N = union.n_points()
        X = union.data()[:, :2]
        X = np.concatenate((X, np.ones((N, 1))), axis=1)

        # Project points to clip space
        Y = X.dot(projection_view)

        Y_visible = np.abs(Y).max(axis=1) <= 1
        visible_idcs = np.where(Y_visible == True)[0]
        invisible_idcs = np.where(Y_visible == False)[0]

        return visible_idcs, invisible_idcs

    def current_union(self, old_or_new='new'):
        return self.current_view().union(old_or_new)


class Timer(QTimer):

    # Emits value between 0 and 1, indicating how far the timer is w.r.t. the total time
    tick = pyqtSignal(float)
    stopped = pyqtSignal()

    def __init__(self, n_steps):
        super().__init__()
        self._n_steps = n_steps
        self._steps = 0
        self.timeout.connect(self.step)

    # Starts a timer, that will give self._n_steps ticks in total_time milliseconds.
    def start(self, total_time, forward=True):
        self._forward = forward
        self._steps = 0
        if self.isActive():
            print('Warning: Tried to start active timer.')
            return
        else:
            super().start(total_time / self._n_steps)

    def step(self):
        self._steps += 1

        if self._forward:
            self.tick.emit(self._steps / self._n_steps)
        else:
            self.tick.emit(1.0 - self._steps / self._n_steps)

        if self._steps == self._n_steps:
            self.stop()
            self.stopped.emit()
