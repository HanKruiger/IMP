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
        input_hidden_features = [None for _ in range(len(self.input_data))]
        for i, input_widget in enumerate(self.input_data.values()):
            dataset = input_widget.get_dataset()
            input_data[i] = dataset
            input_hidden_features[i] = input_widget.hidden_features()
        input_data = tuple(input_data)
        input_hidden_features = tuple(input_hidden_features)

        parameters = {}
        for name, (input_widget, data_type) in self.parameters.items():
            if data_type == int or data_type == float:
                if input_widget.hasAcceptableInput():
                    parameters[name] = data_type(input_widget.text())
            elif data_type == bool:
                parameters[name] = input_widget.isChecked()

        operator = operator_class()
        operator.set_parameters(parameters)
        operator.set_inputs(input_data, input_hidden_features)

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

        for name, (data_type, select_features) in operator_class.input_description().items():
            input_dataset_selector = InputDatasetSelector(name, self.imp_app.datasets_widget.datasets(), select_features)
            input_dataset_selector.select(self.dataset)
            self.input_layout.addLayout(input_dataset_selector)
            self.input_data[name] = input_dataset_selector

        self.parameters = dict()
        self.clear_layout(self.parameters_layout)

        for name, (data_type, default) in operator_class.parameters_description().items():
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
            elif data_type == bool:
                check_box = QCheckBox(name)
                check_box.setChecked(default)
                hbox.addWidget(check_box)
                self.parameters[name] = check_box, data_type

            self.parameters_layout.addLayout(hbox)

class InputDatasetSelector(QHBoxLayout):

    def __init__(self, name, datasets, select_features):
        super().__init__()
        self.combobox = QComboBox()
        self.hidden_features_edit = QLineEdit()

        self.combobox = QComboBox()
        for dataset in datasets:
            self.combobox.addItem(dataset.name, dataset)

        self.combobox.currentIndexChanged.connect(self.on_index_change)

        self.addWidget(QLabel(name))
        self.addWidget(self.combobox)
        if select_features:
            self.addWidget(QLabel('Hidden features'))
            self.addWidget(self.hidden_features_edit)

    def select(self, dataset):
        # Set the clicked dataset as initial input choice
        idx = self.combobox.findData(dataset)
        if idx != -1:
            self.combobox.setCurrentIndex(idx)
        self.hidden_features_edit.setText(', '.join([str(hidden_feature) for hidden_feature in dataset.hidden_features()]))

    @pyqtSlot(int)
    def on_index_change(self, index):
        dataset = self.combobox.currentData(role=Qt.UserRole)
        self.hidden_features_edit.setText(', '.join([str(hidden_feature) for hidden_feature in dataset.hidden_features()]))

    def get_dataset(self):
        return self.combobox.currentData(role=Qt.UserRole)
    
    def hidden_features(self):
        input_text = self.hidden_features_edit.text()
        hidden_features = [int(feature) for feature in input_text.split(',') if feature != '']
        return hidden_features
