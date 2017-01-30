from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.embedders import PCAEmbedder, TSNEEmbedder

class ProjectionDialog(QDialog):

    def __init__(self, parent, dataset, imp_app):
        super().__init__(parent)
        self.dataset = dataset
        self.imp_app = imp_app

        self.projection_technique = QComboBox()
        self.projection_technique.addItem('PCA', PCAEmbedder)
        self.projection_technique.addItem('t-SNE', TSNEEmbedder)

        self.projection_technique.currentIndexChanged.connect(self.build_parameters_ui)

        self.parameters_layout = QVBoxLayout()

        project_button = QPushButton('Project')
        project_button.clicked.connect(self.project_and_close)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.projection_technique)
        vbox.addLayout(self.parameters_layout)
        vbox.addWidget(project_button)
        
        self.setLayout(vbox)
        self.build_parameters_ui()

    def project_and_close(self):
        embedder_class = self.projection_technique.currentData(role=Qt.UserRole)

        parameters = {}
        for name, (input_box, data_type) in self.parameters.items():
            if input_box.hasAcceptableInput():
                parameters[name] = data_type(input_box.text())

        embedder = embedder_class(**parameters)
        self.imp_app.statusBar().showMessage('Making layout with {}...'.format(embedder_class.__name__))
        self.dataset.make_embedding(embedder)
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

    def build_parameters_ui(self, index=0):
        embedder_class = self.projection_technique.currentData(role=Qt.UserRole)
        
        self.parameters = dict()

        self.clear_layout(self.parameters_layout)

        for name, data_type, default in embedder_class.parameters():
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(name))
            input_box = QLineEdit()
            self.parameters[name] = input_box, data_type
            hbox.addWidget(input_box)
            if data_type == float:
                input_box.setValidator(QDoubleValidator())
            elif data_type == int:
                input_box.setValidator(QIntValidator())
            input_box.setText(str(default))

            self.parameters_layout.addLayout(hbox)