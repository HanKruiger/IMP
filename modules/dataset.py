from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np

class Dataset(QObject):
    
    # Emitted when (child) embedding is finished
    embedding_finished = pyqtSignal(object)

    def __init__(self, name, parent=None, X=None):
        super().__init__()
        self.name = name
        self.parent = parent
        self.children = set()
        if X is not None:
            self.set_data(X)

    def set_data(self, X):
        self.X = X
        self.N = X.shape[0]
        if len(X.shape) < 2:
            self.m = 1
        else:
            self.m = X.shape[1]

    def make_embedding(self, Embedder):
        self.embedding_worker = Embedder(self.X)
        self.embedding_worker.finished.connect(self.set_embedding)
        self.embedding_worker.start()

    @pyqtSlot()
    def set_embedding(self):
        # Fetch data from worker, and delete it
        Y = self.embedding_worker.Y
        del self.embedding_worker # Your services are no longer needed.
        
        # Prevent cyclic imports..
        from modules.dataset_2d import Dataset2D

        new_child = Dataset2D(self.name + '_em', self, X=Y)
        self.children.add(new_child) # for now
        self.embedding_finished.emit(new_child)

class InputDataset(Dataset):
    # Emitted when input data is loaded
    data_ready = pyqtSignal()

    def __init__(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        super().__init__(name, parent=None, X=None)

        # Load the data in a separate thread, so the GUI doesn't hang.
        # (Doesn't start reading yet. Still needs to be connected from
        # outside.)
        self.worker = DataLoadWorker(path)
        # Somehow I need this function, and cannot call the method directly...
        def wrapper():
            self.on_data_loaded()
        self.worker.finished.connect(wrapper)

    def load_data(self):
        # Start the thread (calls the run() method in the other thread)
        self.worker.start()

    @pyqtSlot()
    def on_data_loaded(self):
        # Fetch data from worker, and delete it
        X = self.worker.data
        del self.worker # Your services are no longer needed.
        
        # Set the data.
        super().set_data(X)
        
        # Tell that we're ready!
        self.data_ready.emit()

# Worker class that loads the data in a separate thread
class DataLoadWorker(QThread):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        # Make numpy load the data
        self.data = np.loadtxt(self.path)
