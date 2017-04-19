from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *

import numpy as np

from widgets import GLWidget, VisualsWidget, DatasetsWidget


class IMPWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.show()
        self.init_ui()

    def init_ui(self):
        self.gl_widget = GLWidget(self)
        self.setCentralWidget(self.gl_widget)
        toolbar = self.addToolBar('Toolbar')

        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('q')
        quit_action.setStatusTip('Quit application')
        quit_action.triggered.connect(qApp.quit)
        toolbar.addAction(quit_action)

        self.datasets_widget = DatasetsWidget(imp_window=self)
        dataset_bar = QToolBar('Datasets')
        dataset_bar.addWidget(self.datasets_widget)
        self.addToolBar(Qt.LeftToolBarArea, dataset_bar)

        visual_options_bar = QToolBar('Visual options')
        self.addToolBar(Qt.RightToolBarArea, visual_options_bar)

        series = QLineSeries()
        series.append(0, 6)
        series.append(2, 4)
        series.append(3, 8)
        series.append(7, 4)
        series.append(10, 5)
        chart = QChart()
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setTitle('Line chart')
        plot = QChartView(chart)
        plot.setRenderHint(QPainter.Antialiasing)
        stats_bar = QToolBar('Statistics')
        stats_bar.addWidget(plot)
        self.addToolBar(Qt.BottomToolBarArea, stats_bar)

        self.visuals_widget = VisualsWidget(imp_window=self)
        visual_options_bar.addWidget(self.visuals_widget)

        self.center()
        self.setWindowTitle('IMP: Interactive Multiscale Projections')

    def center(self):
        rect = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center_point)
        self.move(rect.topLeft())

    def vis_params(self):
        return self.visuals_widget