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
        self.init_ui()
        self.datasets = []        

    def init_ui(self):
        self.gl_widget = ImpOpenGLWidget(self)
        self.setCentralWidget(self.gl_widget)
        self.setAcceptDrops(True);
        toolbar = self.addToolBar('Toolbar')

        test_button = QAction('Test', self)
        test_button.triggered.connect(self.do_something)
        toolbar.addAction(test_button)
        
        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('q')
        quit_action.setStatusTip('Quit application')
        quit_action.triggered.connect(qApp.quit)
        toolbar.addAction(quit_action)

        sidebar = QToolBar('Sidebar')
        self.addToolBar(Qt.LeftToolBarArea, sidebar)

        pointsize_slider = QSlider(Qt.Horizontal)
        pointsize_slider.setMinimum(1.0)
        pointsize_slider.setMaximum(10.0)
        pointsize_slider.valueChanged.connect(self.gl_widget.set_pointsize)
        pointsize_slider.setValue(8.0)
        sidebar.addWidget(QLabel('Point size'))
        sidebar.addWidget(pointsize_slider)

        self.center()
        self.setWindowTitle('IMP: Interactive Multidimensional Projections')
        self.statusBar().showMessage('Built user interface.', msecs=2000)

    def do_something(self):
        Y = np.random.random((10000, 2))
        self.gl_widget.makeCurrent()
        self.gl_widget.add_object(Dataset2D(Y))
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
            dmd.data_loaded.connect(callback)
            dmd.load()
            self.datasets.append(dmd)

    def center(self):
        rect = self.frameGeometry()
        # No argument: Default screen (for if you use virtual screens)
        center_point = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center_point)
        self.move(rect.topLeft())
