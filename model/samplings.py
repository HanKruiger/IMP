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
            idcs_in_parent = np.random.choice(self.parent.n_points(), self.n_samples, replace=False)
            idcs_in_parent.sort()
            self.ready.emit(idcs_in_parent)

    def __init__(self, parent, n_samples, name=None, hidden=None):
        if name is None:
            name = 'RndS({})'.format(parent.name())

        super().__init__(parent, idcs=None, name=name, hidden=hidden)

        sampler = RandomSampling.RandomSamplingWorker(parent, n_samples)
        self.spawn_thread(sampler, self.set_indices_in_parent, waitfor=(parent,))
