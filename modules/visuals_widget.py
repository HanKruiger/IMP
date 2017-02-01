from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import Dataset

import numpy as np


class VisualsWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Visual attributes')
        self.imp_app = imp_app
        vbox_main = QVBoxLayout()
        self.setLayout(vbox_main)

        pointsize_slider = QSlider(Qt.Horizontal)
        pointsize_slider.setMinimum(1)
        pointsize_slider.setMaximum(30)
        pointsize_slider.valueChanged.connect(self.imp_app.gl_widget.set_pointsize)
        pointsize_slider.setValue(8)
        vbox_main.addWidget(QLabel('Point size'))
        vbox_main.addWidget(pointsize_slider)

        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(255)
        opacity_slider.valueChanged.connect(self.imp_app.gl_widget.set_opacity)
        opacity_slider.setValue(25)
        vbox_main.addWidget(QLabel('Opacity'))
        vbox_main.addWidget(opacity_slider)

        self.attributes = {
            'position': AttributeComboBox('position', self.imp_app),
            'color': AttributeComboBox('color', self.imp_app)
        }

        self.attribute_datasets_model = QStandardItemModel()
        self.attribute_datasets_model.appendRow(QStandardItem(None))

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

    def use_as_attribute(self, dataset):
        item = QStandardItem(dataset.name)
        item.setData(dataset, role=Qt.UserRole)
        self.attribute_datasets_model.appendRow(item)

    def is_in_attributes(self, dataset):
        for idx in range(self.attribute_datasets_model.rowCount()):
            item = self.attribute_datasets_model.item(idx)
            if item is not None and item.data(role=Qt.UserRole) == dataset:
                return True
        return False

    def remove_as_attribute(self, dataset):
        for idx in range(self.attribute_datasets_model.rowCount()):
            item = self.attribute_datasets_model.item(idx)
            if item is not None and item.data(role=Qt.UserRole) == dataset:
                succeeded = self.attribute_datasets_model.removeRow(idx)
                if succeeded:
                    dataset.destroy_vbo()
                    return
                else:
                    print('Failed to remove attribute!')

    def clear_attributes(self):
        for idx in reversed(range(self.attribute_datasets_model.rowCount())):
            item = self.attribute_datasets_model.item(idx)
            if item is not None:
                dataset = item.data(role=Qt.UserRole)
                if isinstance(dataset, Dataset):
                    succeeded = self.attribute_datasets_model.removeRow(idx)
                    if succeeded:
                        dataset.destroy_vbo()
                    else:
                        print('Failed to remove attribute!')


class AttributeComboBox(QComboBox):

    def __init__(self, attribute, imp_app):
        super().__init__()
        self.attribute = attribute
        self.imp_app = imp_app

    @pyqtSlot(int)
    def set_attribute(self, index):
        dataset = self.model().item(index).data(role=Qt.UserRole)
        if dataset is None:
            self.imp_app.gl_widget.disable_attribute(self.attribute)
        else:
            self.imp_app.gl_widget.set_attribute(dataset.vbo(), dataset.N, dataset.m, self.attribute)
