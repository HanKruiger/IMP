from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

from modules.impopenglwidget import ImpOpenGLWidget
from modules.dataset2d import Dataset2D

class ImpApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()

        Y = np.random.random((10000, 2)) * 2 - 1
        self.d2d = Dataset2D(Y)
        self.gl_widget.add_object(self.d2d)


    def init_ui(self):
        self.gl_widget = ImpOpenGLWidget()
        self.setCentralWidget(self.gl_widget)

        self.toolbar = self.addToolBar('Main toolbar')

        change_action = QAction('Change', self)
        change_action.triggered.connect(self.do_something)
        self.toolbar.addAction(change_action)
        
        quit_action = QAction('&Quit', self)  # The '&' signifies the via-menu keyboard shortcut (Alt+E in this case)
        quit_action.setShortcut('q')
        quit_action.setStatusTip('Quit application')
        quit_action.triggered.connect(qApp.quit)
        self.toolbar.addAction(quit_action)

        self.center()
        self.setWindowTitle('IMP: Interactive Multidimensional Projections')

        self.show()

    def do_something(self):
        Y = np.random.random((np.random.choice([1000000, 1000]), 2)) * 2 - 1
        self.d2d.update_Y(Y)

        # Schedule redraw
        self.gl_widget.update()

    def center(self):
        rect = self.frameGeometry()
        # No argument: Default screen (for if you use virtual screens)
        center_point = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center_point)
        self.move(rect.topLeft())
