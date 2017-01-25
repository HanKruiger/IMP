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
        self.visual_attributes_widget = VisualAttributesWidget(imp_app=self)
        visual_options_bar.addWidget(self.visual_attributes_widget)

        self.center()
        self.setWindowTitle('IMP: Interactive Multiscale Projections')
        self.statusBar().showMessage('Built user interface.', msecs=2000)

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
        self.id_max = 0

        self.id_items = dict()
        self.datasets = dict()
        self.tree_view = QTreeView()
        self.model = QStandardItemModel()
        self.tree_view.setModel(self.model)
        self.model.setHorizontalHeaderLabels(['ID', 'Name', 'N', 'm'])
        self.vbox_main.addWidget(self.tree_view)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_menu)
        
        self.setAcceptDrops(True)


    @pyqtSlot(object)
    def add_dataset(self, dataset):
        print(dataset.name)
        name_item = QStandardItem(dataset.name)
        N_item = QStandardItem(str(dataset.N))
        m_item = QStandardItem(str(dataset.m))
        
        the_id = str(self.id_max)
        id_item = QStandardItem(the_id)
        self.id_max += 1

        self.id_items[dataset] = id_item
        self.datasets[the_id] = dataset
        if dataset.parent() is None:
            self.model.appendRow([id_item, name_item, N_item, m_item])
        else:
            self.id_items[dataset.parent()].appendRow([id_item, name_item, N_item, m_item])
        for i in range(4):
            self.tree_view.resizeColumnToContents(i)

    def open_menu(self, position):
        indexes = self.tree_view.selectedIndexes()
        if len(indexes) != 4:
            return

        # Get the id of the selected row
        the_id = self.model.data(indexes[0])

        # Get the Dataset object by using the id
        dataset = self.datasets[the_id]

        # Build menu, based on characteristics of Dataset object.
        menu = QMenu()

        if type(dataset) == Dataset2D:
            if dataset in self.imp_app.gl_widget.objects:
                unview_action = menu.addAction('Unview')
                @pyqtSlot()
                def unview_embedding():
                    self.imp_app.gl_widget.remove_object(dataset)
                unview_action.triggered.connect(unview_embedding)
            else:
                view_action = menu.addAction('View')

                @pyqtSlot()
                def view_embedding():
                    self.imp_app.gl_widget.add_object(dataset)

                view_action.triggered.connect(view_embedding)
        if type(dataset) == Dataset or type(dataset) == InputDataset:
            embed_action = menu.addAction('PCA to 2D')

            @pyqtSlot()
            def make_embedding():
                dataset.embedding_finished.connect(self.add_dataset)
                dataset.make_embedding(PCAEmbedder(n_components=2))

            embed_action.triggered.connect(make_embedding)
        
        menu.exec_(self.tree_view.viewport().mapToGlobal(position))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            if not all([url.isValid() for url in urls]):
                qDebug('Invalid URL(s): {0}'.format([url.toString() for url in urls if not url.isValid()]))
            elif not all([url.isLocalFile() for url in urls]):
                qDebug('Non-local URL(s): {0}'.format([url.toString() for url in urls if not url.isLocalFile()]))
            else:
                self.imp_app.statusBar().showMessage('Drop to load {0} as new dataset'.format(', '.join([url.fileName() for url in urls])))
                e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.imp_app.statusBar().clearMessage()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        for url in urls:
            input_dataset = InputDataset(url.path())

            @pyqtSlot()
            def callback():
                self.imp_app.statusBar().clearMessage()
                self.add_dataset(input_dataset)

            input_dataset.data_ready.connect(callback)
            self.imp_app.statusBar().showMessage('Loading {0}...'.format(url.fileName()))
            input_dataset.load_data()

    def show_hint(self, message):
        self.drop_hint = QLabel(message)
        self.vbox_main.addWidget(self.drop_hint)

    def hide_hint(self):
        self.vbox_main.removeWidget(self.drop_hint)
        del self.drop_hint

class VisualAttributesWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Visual attributes')
        self.imp_app = imp_app
        self.vbox_main = QVBoxLayout()
        self.setLayout(self.vbox_main)
