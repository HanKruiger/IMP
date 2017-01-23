from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np

class DatasetMD(QObject):
    data_loaded = pyqtSignal()
    def __init__(self, path):
        super().__init__()
        self.name = os.path.splitext(os.path.basename(path))[0]
        
        # Load the data in a separate thread, so the GUI doesn't hang.
        self.io_thread = IOThread(path)
        self.io_thread.data_loaded.connect(self.set_data)

    def load(self):
        self.io_thread.start()

    def set_data(self, data):
        self.data = data
        self.data_loaded.emit()

class IOThread(QThread):
    data_loaded = pyqtSignal(np.ndarray)

    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        self.data = np.loadtxt(self.path)
        self.data_loaded.emit(self.data)
