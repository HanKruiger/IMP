from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *
import numpy as np

class RandomSampling(Selection):

    class RandomSamplingWorker(QObject):

        ready = pyqtSignal(object)

        def __init__(self, parent, n_samples):
            super().__init__()
            self.parent = parent
            self.n_samples = n_samples

        def work(self):
            idcs = np.random.choice(self.parent.n_points(), self.n_samples, replace=False)
            self.ready.emit(idcs)

    def __init__(self, parent, n_samples, name=None, hidden=None):
        if name is None:
            name = 'RndS({})'.format(parent.name())

        super().__init__(parent, idcs=None, name=name, hidden=hidden)

        sampler = RandomSampling.RandomSamplingWorker(parent, n_samples)
        self.spawn_thread(sampler, self.set_indices, waitfor=(parent,))
