from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.clusterers import *


class ClusteringDialog(QDialog):

    def __init__(self, parent, dataset, imp_app):
        super().__init__(parent)
        self.dataset = dataset
        self.imp_app = imp_app

        self.clustering_technique = QComboBox()
        self.clustering_technique.addItem('KMeans', KMeansClusterer)
        self.clustering_technique.addItem('MiniBatchKMeans', MiniBatchKMeansClusterer)

        self.clustering_technique.currentIndexChanged.connect(self.build_parameters_ui)

        self.parameters_layout = QVBoxLayout()

        cluster_button = QPushButton('Cluster')
        cluster_button.clicked.connect(self.cluster_and_close)

        vbox = QVBoxLayout()
        vbox.addWidget(self.clustering_technique)
        vbox.addLayout(self.parameters_layout)
        vbox.addWidget(cluster_button)

        self.setLayout(vbox)
        self.build_parameters_ui()

    def cluster_and_close(self):
        clusterer_class = self.clustering_technique.currentData(role=Qt.UserRole)

        parameters = {}
        for name, (input_box, data_type) in self.parameters.items():
            if input_box.hasAcceptableInput():
                parameters[name] = data_type(input_box.text())

        clusterer = clusterer_class()
        clusterer.set_parameters(parameters)

        self.imp_app.statusBar().showMessage('Making clustering with {}...'.format(clusterer_class.__name__))
        self.dataset.make_embedding(clusterer)
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
        clusterer_class = self.clustering_technique.currentData(role=Qt.UserRole)

        self.parameters = dict()

        self.clear_layout(self.parameters_layout)

        for name, data_type, default in clusterer_class.parameters_description():
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
