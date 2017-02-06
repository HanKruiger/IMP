from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.embedders import *
from modules.clusterers import *
from modules.linalg_ops import *


class OperatorDialog(QDialog):

    def __init__(self, parent, dataset, imp_app):
        super().__init__(parent)
        self.dataset = dataset
        self.imp_app = imp_app

        self.operators = QComboBox()
        self.operators.addItem('PCA', PCAEmbedder)
        self.operators.addItem('t-SNE', TSNEEmbedder)
        self.operators.addItem('LLE', LLEEmbedder)
        self.operators.addItem('Spectral', SpectralEmbedder)
        self.operators.addItem('MDS', MDSEmbedder)
        self.operators.addItem('Isomap', IsomapEmbedder)
        self.operators.addItem('MiniBatchKMeans', MiniBatchKMeansClusterer)
        self.operators.addItem('KMeans', KMeansClusterer)
        self.operators.addItem('HorzCat', HorizontalConcat)

        self.operators.currentIndexChanged.connect(self.build_ui)

        self.input_layout = QVBoxLayout()
        self.parameters_layout = QVBoxLayout()

        run_button = QPushButton('Run')
        run_button.clicked.connect(self.run_and_close)

        vbox = QVBoxLayout()
        vbox.addWidget(self.operators)
        vbox.addLayout(self.input_layout)
        vbox.addLayout(self.parameters_layout)
        vbox.addWidget(run_button)

        self.setLayout(vbox)
        self.build_ui()

    def run_and_close(self):
        operator_class = self.operators.currentData(role=Qt.UserRole)

        input_data = [None for _ in range(len(self.input_data))]
        for i, (name, (input_widget, data_type)) in enumerate(self.input_data.items()):
            if data_type == Dataset:
                input_data[i] = input_widget.currentData(role=Qt.UserRole)
        input_data = tuple(input_data)

        parameters = {}
        for name, (input_widget, data_type) in self.parameters.items():
            if data_type == int or data_type == float:
                if input_widget.hasAcceptableInput():
                    parameters[name] = data_type(input_widget.text())
            elif data_type == Dataset:
                parameters[name] = input_widget.currentData(role=Qt.UserRole)

        operator = operator_class()
        operator.set_parameters(parameters)
        operator.set_inputs(input_data)

        self.imp_app.statusBar().showMessage('Performing operation with {}...'.format(operator_class.__name__))
        input_data[0].perform_operation(operator)
        self.done(0)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def build_ui(self, index=0):
        operator_class = self.operators.currentData(role=Qt.UserRole)

        self.input_data = dict()
        self.clear_layout(self.input_layout)

        for name, data_type in operator_class.input_description():
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(name))
            combobox = QComboBox()
            for dataset in self.imp_app.datasets_widget.datasets():
                combobox.addItem(dataset.name, dataset)

            # Set the clicked dataset as default input for all inputs.
            idx = combobox.findData(self.dataset)
            if idx != -1:
                combobox.setCurrentIndex(idx)

            hbox.addWidget(combobox)
            self.input_data[name] = combobox, data_type
            self.input_layout.addLayout(hbox)

        self.parameters = dict()
        self.clear_layout(self.parameters_layout)

        for name, data_type, default in operator_class.parameters_description():
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(name))
            if data_type == float or data_type == int:
                input_box = QLineEdit()
                if data_type == float:
                    input_box.setValidator(QDoubleValidator())
                elif data_type == int:
                    input_box.setValidator(QIntValidator())
                input_box.setText(str(default))

                hbox.addWidget(input_box)
                self.parameters[name] = input_box, data_type
            elif data_type == Dataset:
                combobox = QComboBox()
                for dataset in self.imp_app.datasets_widget.datasets():
                    combobox.addItem(dataset.name, dataset)

                hbox.addWidget(combobox)
                self.parameters[name] = combobox, data_type

            self.parameters_layout.addLayout(hbox)
