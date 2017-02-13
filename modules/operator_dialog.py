from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.embedders import *
from modules.clusterers import *
from modules.linalg_ops import *
from modules.samplers import *


class OperatorDialog(QDialog):

    def __init__(self, parent, dataset, imp_app):
        super().__init__(parent)
        self.dataset = dataset
        self.imp_app = imp_app

        self.operators = QComboBox()
        self.operators.addItem('PCA', PCAEmbedder)
        self.operators.addItem('Random sampler', RandomSampler)
        self.operators.addItem('LAMP', LAMPEmbedder)
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

        inputs = {}
        for name, input_widget in self.input_form.items():
            inputs[name] = input_widget.value()

        parameters = {}
        for name, (input_widget, data_type) in self.parameter_form.items():
                # def hidden_features(self):
            if name == 'hidden_features':
                input_text = input_widget.text()
                parameters[name] = [int(feature) for feature in input_text.split(',') if feature != '']
            elif data_type == int or data_type == float:
                if input_widget.hasAcceptableInput():
                    parameters[name] = data_type(input_widget.text())
            elif data_type == bool:
                parameters[name] = input_widget.isChecked()

        operator = operator_class()
        operator.set_parameters(parameters)
        operator.set_input(inputs)

        self.imp_app.statusBar().showMessage('Performing operation with {}...'.format(operator_class.__name__))
        inputs['parent'].perform_operation(operator)
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

        self.input_form = dict()
        self.clear_layout(self.input_layout)

        for name, description in operator_class.input_description().items():
            input_dataset_selector = InputDatasetSelector(name, self.imp_app.datasets_widget.datasets())
            input_dataset_selector.select(self.dataset)
            self.input_layout.addLayout(input_dataset_selector)
            self.input_form[name] = input_dataset_selector
                

        self.parameter_form = dict()
        self.clear_layout(self.parameters_layout)

        for name, (data_type, default) in operator_class.parameters_description().items():
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(name))
            if name == 'hidden_features':
                hf_text_box = QLineEdit()
                def set_default_hidden_features():
                    dataset = self.input_form['parent'].value()
                    hf_text_box.setText(', '.join([str(hidden_feature) for hidden_feature in dataset.hidden_features()]))
                set_default_hidden_features()
                self.input_form['parent'].combobox.currentIndexChanged.connect(set_default_hidden_features)
                hbox.addWidget(hf_text_box)
                self.parameter_form[name] = hf_text_box, data_type
            elif data_type == float or data_type == int:
                text_box = QLineEdit()
                if data_type == float:
                    text_box.setValidator(QDoubleValidator())
                elif data_type == int:
                    text_box.setValidator(QIntValidator())
                text_box.setText(str(default))

                hbox.addWidget(text_box)
                self.parameter_form[name] = text_box, data_type
            elif data_type == bool:
                check_box = QCheckBox(name)
                check_box.setChecked(default)
                hbox.addWidget(check_box)
                self.parameter_form[name] = check_box, data_type

            self.parameters_layout.addLayout(hbox)

class InputDatasetSelector(QHBoxLayout):

    def __init__(self, name, datasets):
        super().__init__()
        self.combobox = QComboBox()

        for dataset in datasets:
            self.combobox.addItem(dataset.name(), dataset)

        self.addWidget(QLabel(name))
        self.addWidget(self.combobox)

    def select(self, dataset):
        # Set the clicked dataset as initial input choice
        idx = self.combobox.findData(dataset)
        if idx != -1:
            self.combobox.setCurrentIndex(idx)

    def value(self):
        dataset = self.combobox.currentData(role=Qt.UserRole)
        return dataset
