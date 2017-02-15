from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from collections import OrderedDict
import time

import numpy as np


class VisualsWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Visual attributes')
        self.imp_app = imp_app
        vbox_main = QVBoxLayout()
        self.setLayout(vbox_main)
        self._current_dataset = None

        pointsize_slider = QSlider(Qt.Horizontal)
        pointsize_slider.setMinimum(1)
        pointsize_slider.setMaximum(500)
        pointsize_slider.valueChanged.connect(self.imp_app.gl_widget.set_pointsize)
        pointsize_slider.setValue(16)
        vbox_main.addWidget(QLabel('Point size'))
        vbox_main.addWidget(pointsize_slider)

        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(255)
        opacity_slider.valueChanged.connect(self.imp_app.gl_widget.set_opacity)
        opacity_slider.setValue(128)
        vbox_main.addWidget(QLabel('Opacity'))
        vbox_main.addWidget(opacity_slider)

        self.attributes = OrderedDict()
        self.attributes['position_x'] = AttributeComboBox('position_x', self, self.imp_app)
        self.attributes['position_y'] = AttributeComboBox('position_y', self, self.imp_app)
        self.attributes['color'] = AttributeComboBox('color', self, self.imp_app)

        self.attribute_datasets_model = QStandardItemModel()
        empty_item = QStandardItem('Pick one')
        empty_item.setData(-1, role=Qt.UserRole)
        self.attribute_datasets_model.appendRow(empty_item)

        clear_button = QPushButton('Clear')
        clear_button.clicked.connect(self.clear_attributes)
        vbox_main.addWidget(clear_button)

        for attribute, combo_box in self.attributes.items():
            combo_box.setModel(self.attribute_datasets_model)
            combo_box.currentIndexChanged.connect(combo_box.set_attribute)

            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(attribute))
            hbox.addWidget(combo_box)
            vbox_main.addLayout(hbox)

    def update_attribute_list(self, dataset):
        self.clear_attributes()
        self._current_dataset = dataset

        for dim in range(dataset.m):
            item = QStandardItem('{}:{}'.format(dataset.name(), dim))
            item.setData(dim, role=Qt.UserRole)
            self.attribute_datasets_model.appendRow(item)

        for i, acb in enumerate(self.attributes.values()):
            if i >= dataset.m:
                break
            acb.setCurrentIndex(i + 1) # Skip the empty entry.

    def current_dataset(self):
        return self._current_dataset

    def clear_attributes(self):
        for idx in reversed(range(self.attribute_datasets_model.rowCount())):
            item = self.attribute_datasets_model.item(idx)
            if item is not None:
                if item.data(role=Qt.UserRole) != -1:
                    succeeded = self.attribute_datasets_model.removeRow(idx)
                    if not succeeded:
                        print('Failed to remove attribute!')

        if self.current_dataset() is not None:
            self.current_dataset().destroy_vbos()
            self._current_dataset = None


class AttributeComboBox(QComboBox):

    def __init__(self, attribute, v_widget, imp_app):
        super().__init__()
        self.attribute = attribute
        self.imp_app = imp_app
        self.v_widget = v_widget

    @pyqtSlot(int)
    def set_attribute(self, index):
        item = self.model().item(index)
        dim = item.data(role=Qt.UserRole)
        dataset = self.v_widget.current_dataset()

        self.imp_app.gl_widget.makeCurrent()
        if dim == -1:
            self.imp_app.gl_widget.disable_attribute(self.attribute)
        else:
            self.imp_app.gl_widget.set_attribute(dataset, dim, dataset.N, 1, self.attribute)
        self.imp_app.gl_widget.doneCurrent()
