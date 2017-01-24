from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

from modules.impopenglwidget import ImpOpenGLWidget
from modules.dataset_2d import Dataset2D
from modules.dataset import Dataset
from modules.dataset import InputDataset
from modules.embedders import PCAEmbedder

class ImpApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.show()
        self.init_ui()

    def init_ui(self):
        self.gl_widget = ImpOpenGLWidget(self)
        self.setCentralWidget(self.gl_widget)
        self.setAcceptDrops(True);
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

        pointsize_slider = QSlider(Qt.Horizontal)
        pointsize_slider.setMinimum(1.0)
        pointsize_slider.setMaximum(10.0)
        pointsize_slider.valueChanged.connect(self.gl_widget.set_pointsize)
        pointsize_slider.setValue(8.0)
        visual_options_bar.addWidget(QLabel('Point size'))
        visual_options_bar.addWidget(pointsize_slider)

        self.center()
        self.setWindowTitle('IMP: Interactive Multiscale Projections')
        self.statusBar().showMessage('Built user interface.', msecs=2000)

    def dragLeaveEvent(self, drag_event):
        self.statusBar().clearMessage()

    def dragEnterEvent(self, drag_event):
        if drag_event.mimeData().hasUrls():
            urls = drag_event.mimeData().urls()
            if not all([url.isValid() for url in urls]):
                qDebug('Invalid URL(s): {0}'.format([url.toString() for url in urls if not url.isValid()]))
            elif not all([url.isLocalFile() for url in urls]):
                qDebug('Non-local URL(s): {0}'.format([url.toString() for url in urls if not url.isLocalFile()]))
            else:
                self.statusBar().showMessage('Drop to load {0}'.format(', '.join([url.fileName() for url in urls])))
                drag_event.acceptProposedAction()

    def dropEvent(self, drop_event):
        urls = drop_event.mimeData().urls()
        for url in urls:
            input_dataset = InputDataset(url.path())

            @pyqtSlot()
            def callback():
                self.statusBar().clearMessage()
                self.datasets_widget.add_dataset(input_dataset)

            input_dataset.data_ready.connect(callback)
            self.statusBar().showMessage('Loading {0}...'.format(url.fileName()))
            input_dataset.load_data()

    def center(self):
        rect = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center_point)
        self.move(rect.topLeft())

class DatasetsWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Datasets')
        self.imp_app = imp_app
        self.vbox_main = QVBoxLayout()
        self.setLayout(self.vbox_main)

    @pyqtSlot(object)
    def add_dataset(self, dataset):

        hbox = QHBoxLayout()

        # Add some labels
        hbox.addWidget(QLabel(dataset.name))
        hbox.addWidget(QLabel(str(dataset.N)))
        hbox.addWidget(QLabel(str(dataset.m)))

        # Add a button, based on the dimensionality of the dataset.
        if dataset.m > 2:
            embed_button = QPushButton(text='Embed')
            def embed_dataset():
                dataset.make_embedding(PCAEmbedder)
            embed_button.clicked.connect(embed_dataset)
            dataset.embedding_finished.connect(self.add_dataset)
            hbox.addWidget(embed_button)
        elif dataset.m == 2:
            visibility_button = QPushButton(text='View')
            def toggle_visibility():
                if dataset in self.imp_app.gl_widget.objects:
                    self.imp_app.gl_widget.remove_object(dataset)
                    visibility_button.setText('View')
                else:
                    self.imp_app.gl_widget.add_object(dataset)
                    visibility_button.setText('Unview')
                self.imp_app.gl_widget.update()
            visibility_button.clicked.connect(toggle_visibility)
            hbox.addWidget(visibility_button)

        # Add it to the main layout
        self.vbox_main.addLayout(hbox)
