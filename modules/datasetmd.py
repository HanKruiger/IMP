from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.pca_embedder import PCAEmbedder
from modules.dataset2d import Dataset2D

import os
import numpy as np

class DatasetMD(QObject):
    data_loaded = pyqtSignal()
    embedding_finished = pyqtSignal(Dataset2D)

    def __init__(self, path):
        super().__init__()
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.embedding_thread = None
        self.embedding = None
        
        # Load the data in a separate thread, so the GUI doesn't hang.
        self.io_thread = IOThread(path)
        self.io_thread.data_loaded.connect(self.set_data)

    def load_data(self):
        self.io_thread.start()

    def set_data(self, X):
        self.X = X
        self.N = X.shape[0]
        if len(X.shape) < 2:
            self.m = 1
        else:
            self.m = X.shape[1]
        self.data_loaded.emit()

    def make_embedding(self):
        self.embedding_thread = PCAEmbedder(self.X)
        self.embedding_thread.embedding_finished.connect(self.set_embedding)
        self.embedding_thread.start()

    def set_embedding(self, Y):
        self.embedding = Dataset2D(Y)
        self.embedding_finished.emit(self.embedding)

class IOThread(QThread):
    data_loaded = pyqtSignal(np.ndarray)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        self.data = np.loadtxt(self.path)
        self.data_loaded.emit(self.data)
