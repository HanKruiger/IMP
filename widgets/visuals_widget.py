from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class VisualsWidget(QGroupBox):

    def __init__(self, imp_window):
        super().__init__('Visual parameters')
        self.imp_window = imp_window
        vbox_main = QVBoxLayout()
        self.setLayout(vbox_main)

        self.sliders = dict()

        self.sliders['point_size'] = Slider(1, 20, 8)
        vbox_main.addWidget(QLabel('Point size'))
        vbox_main.addWidget(self.sliders['point_size'])
        self.sliders['point_size'].valueChanged.connect(self.imp_window.gl_widget.update)

        self.sliders['opacity'] = Slider(0, 1, 0.6)
        self.sliders['opacity'].valueChanged.connect(self.imp_window.gl_widget.update)
        vbox_main.addWidget(QLabel('Opacity'))
        vbox_main.addWidget(self.sliders['opacity'])

        self.sliders['new_points_interpolation'] = Slider(0, 1, 1)
        self.sliders['new_points_interpolation'].valueChanged.connect(self.imp_window.gl_widget.update)
        vbox_main.addWidget(QLabel('Interpolation debugging'))
        vbox_main.addWidget(self.sliders['new_points_interpolation'])
    
    def get(self, name):
        slider = self.sliders[name]
        return slider.get()

    def set(self, name, value):
        slider = self.sliders[name]
        slider.set(value)

class Slider(QSlider):

    def __init__(self, mini=0, maxi=1, default_value=0, direction=Qt.Horizontal, n_steps=20):
        super().__init__(direction)
        self.setMinimum(0)
        self.setMaximum(n_steps)
        self._maxi = maxi
        self._mini = mini
        self.set(default_value)

    def set(self, value):
        self.setValue(self.minimum() + (value - self._mini) / (self._maxi - self._mini) * (self.maximum() - self.minimum()))

    def get(self):
        return self._mini + ((self.value() - self.minimum()) / (self.maximum() - self.minimum())) * (self._maxi - self._mini)