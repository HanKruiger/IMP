from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

from modules.impopenglwidget import ImpOpenGLWidget
from modules.dataset2d import Dataset2D
from modules.datasetmd import DatasetMD

class ImpApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.show()
        self.selected_dataset = None
        self.datasets = []        
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

        self.datasets_widget = DatasetsWidget(self.datasets)
        dataset_bar = QToolBar('Datasets')
        self.addToolBar(Qt.LeftToolBarArea, dataset_bar)
        dataset_bar.addWidget(self.datasets_widget)
        
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
        self.setWindowTitle('IMP: Interactive Multidimensional Projections')
        self.statusBar().showMessage('Built user interface.', msecs=2000)

    def select_dataset(self, dataset):
        self.selected_dataset = dataset
        dataset.embedding_finished.connect(self.bind_embedding)
        self.statusBar().showMessage('Making embedding...')
        dataset.make_embedding()

    def deselect_dataset(self, embedding=None):
        if embedding is not None:
            self.gl_widget.remove_object(embedding)
        self.selected_dataset = None

    def bind_embedding(self, embedding):
        self.statusBar().showMessage('Binding embedding...')

        self.gl_widget.makeCurrent()
        self.gl_widget.add_object(embedding)
        self.gl_widget.doneCurrent()

        # Schedule redraw
        self.gl_widget.update()

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
            self.statusBar().showMessage('Loading {0}...'.format(url.fileName()))
            dmd = DatasetMD(url.path())
            def callback():
                self.statusBar().showMessage('Done loading {0}.'.format(url.fileName()))
                self.datasets_widget.refresh()
            dmd.data_loaded.connect(callback)
            dmd.load_data()
            self.datasets.append(dmd)

    def center(self):
        rect = self.frameGeometry()
        # No argument: Default screen (for if you use virtual screens)
        center_point = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center_point)
        self.move(rect.topLeft())

# Maintains a list of loaded datasets
class DatasetsWidget(QGroupBox):
    def __init__(self, datasets, parent=None):
        super().__init__('Datasets', parent)
        self.datasets = datasets
        self.init_ui()

    def init_ui(self):
        if self.layout() is not None:
            # An anonymous QWidget adopts the layout, so we can replace it.
            # (It will be deleted by the GC, since it will have no references
            # to it when this function ends.)
            QWidget().setLayout(self.layout())

        grid = QGridLayout()
        for i, dataset in enumerate(self.datasets):
            grid.addWidget(QLabel(dataset.name), i, 0)
            grid.addWidget(QLabel(str(dataset.N)), i, 1)
            grid.addWidget(QLabel(str(dataset.m)), i, 2)
            layout_button = QPushButton(text='Select')
            def select_dataset():
                self.parent().parent().select_dataset(dataset)
            layout_button.clicked.connect(select_dataset)
            # layout_button.clicked.connect(hoi)
            grid.addWidget(layout_button)
        self.setLayout(grid)

    def refresh(self):
        self.init_ui()