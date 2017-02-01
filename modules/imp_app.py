from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

from modules.opengl_widget import OpenGLWidget
from modules.datasets_widget import DatasetsWidget
from modules.visuals_widget import VisualsWidget
from modules.dataset import Dataset
from modules.dataset import DatasetItem
from modules.dataset import InputDataset
from modules.embedders import PCAEmbedder


class ImpApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.show()
        self.init_ui()

    def init_ui(self):
        self.gl_widget = OpenGLWidget(self)
        self.setCentralWidget(self.gl_widget)
        toolbar = self.addToolBar('Toolbar')

        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('q')
        quit_action.setStatusTip('Quit application')
        quit_action.triggered.connect(qApp.quit)
        toolbar.addAction(quit_action)

        self.datasets_widget = DatasetsWidget(imp_app=self)
        dataset_bar = QToolBar('Datasets')
        dataset_bar.addWidget(self.datasets_widget)
        self.addToolBar(Qt.LeftToolBarArea, dataset_bar)

        visual_options_bar = QToolBar('Visual options')
        self.addToolBar(Qt.RightToolBarArea, visual_options_bar)

        self.visuals_widget = VisualsWidget(imp_app=self)
        visual_options_bar.addWidget(self.visuals_widget)

        self.center()
        self.setWindowTitle('IMP: Interactive Multiscale Projections')
        self.statusBar().showMessage('Built user interface.', msecs=2000)

    def center(self):
        rect = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center_point)
        self.move(rect.topLeft())
