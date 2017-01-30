from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class VisualAttributesWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Visual attributes')
        self.imp_app = imp_app
        self.vbox_main = QVBoxLayout()
        self.setLayout(self.vbox_main)
        clear_button = QPushButton('Clear visualization')
        clear_button.clicked.connect(self.clear_visualization)
        self.vbox_main.addWidget(clear_button)

    def clear_visualization(self):
        self.imp_app.gl_widget.clear()
