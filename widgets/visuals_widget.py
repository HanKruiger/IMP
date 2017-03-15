from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class VisualsWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Visual attributes')
        self.imp_app = imp_app
        vbox_main = QVBoxLayout()
        self.setLayout(vbox_main)

        pointsize_slider = QSlider(Qt.Horizontal)
        pointsize_slider.setMinimum(1)
        pointsize_slider.setMaximum(500)
        pointsize_slider.valueChanged.connect(self.imp_app.gl_widget.set_pointsize)
        pointsize_slider.setValue(64)
        vbox_main.addWidget(QLabel('Point size'))
        vbox_main.addWidget(pointsize_slider)

        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(255)
        opacity_slider.valueChanged.connect(self.imp_app.gl_widget.set_opacity)
        opacity_slider.setValue(128)
        vbox_main.addWidget(QLabel('Opacity'))
        vbox_main.addWidget(opacity_slider)
