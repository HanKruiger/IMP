from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from widgets import Slider

class VisualsWidget(QGroupBox):

    def __init__(self, imp_window):
        super().__init__('Visual parameters')
        self.imp_window = imp_window
        vbox_main = QVBoxLayout()
        self.setLayout(vbox_main)

        self.sliders = dict()

        self.sliders['point_size'] = Slider('Point size', 1, 20, 8, data_type=float)
        self.sliders['point_size'].slider.valueChanged.connect(self.imp_window.gl_widget.update)
        vbox_main.addLayout(self.sliders['point_size'])

        self.sliders['opacity_regular'] = Slider('Opacity (regular points)', 0, 1, 0.4, data_type=float)
        self.sliders['opacity_regular'].slider.valueChanged.connect(self.imp_window.gl_widget.update)
        vbox_main.addLayout(self.sliders['opacity_regular'])

        self.sliders['opacity_representatives'] = Slider('Opacity (representatives)', 0, 1, 1, data_type=float)
        self.sliders['opacity_representatives'].slider.valueChanged.connect(self.imp_window.gl_widget.update)
        vbox_main.addLayout(self.sliders['opacity_representatives'])

        self.colour_data = None
        colour_hbox = QHBoxLayout()
        colour_hbox.addWidget(QLabel('Colour'))
        self.colour_combo_box = QComboBox()
        self.colour_combo_box.currentIndexChanged.connect(self.colour_changed)
        colour_hbox.addWidget(self.colour_combo_box)
        vbox_main.addLayout(colour_hbox)
    
    def get(self, name):
        slider = self.sliders[name]
        return slider.value()

    def set(self, name, value):
        slider = self.sliders[name]
        slider.set_value(value)

    def add_colour_option(self, name, data):
        self.colour_combo_box.addItem(name, userData=data)

    @pyqtSlot(int)
    def colour_changed(self, index):
        self.colour_data = self.colour_combo_box.itemData(index, role=Qt.UserRole)
        self.imp_window.gl_widget.dataset_view_renderer.add_colour(self.get_colour())

    def get_colour(self):
        return self.colour_data