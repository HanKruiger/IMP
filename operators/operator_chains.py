from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from model.dataset import Dataset, Merging
from operators.operator import Operator
from operators.embedders import LAMPEmbedder
from operators.samplers import RandomSampler
from operators.selectors import LenseSelector
from operators.fetchers import HypersphereFetcher

class HierarchicalZoom(QObject):

    when_done = pyqtSignal(object)

    def __init__(self, parent, selected, center, radius, x_dim, y_dim):
        super().__init__()
        self.selector = LenseSelector()
        self.selector.set_input({
            'embedding': selected,
            'parent': parent
            })
        self.selector.set_parameters({
            'center': center,
            'radius': radius,
            'x_dim': x_dim,
            'y_dim': y_dim
        })
        self.parent = parent

    def run(self):
        self.parent.operation_finished.connect(self.step_1)
        self.parent.perform_operation(self.selector)

    # Fetch operation
    @pyqtSlot(object, object)
    def step_1(self, result, operator):
        if operator != self.selector:
            return # This was not the operator I was waiting for.
        
        self.query_nd = result[0]
        self.query_2d = result[1]

        # Unsubscribe from further events on dataset
        self.parent.operation_finished.disconnect(self.step_1)

        self.root_dataset = self.parent
        while self.root_dataset.parent() is not None and not isinstance(self.root_dataset, Merging):
            self.root_dataset = self.root_dataset.parent()

        self.fetcher = HypersphereFetcher()
        self.fetcher.set_input({
            'nd_dataset': self.root_dataset,
            'query_nd': self.query_nd,
            'query_2d': self.query_2d
        })
        self.fetcher.set_parameters({
            'center': self.selector.parameters()['center'],
            'radius': self.selector.parameters()['radius']    
        })

        self.root_dataset.operation_finished.connect(self.step_2)
        self.root_dataset.perform_operation(self.fetcher)
        
    # Subsample operation, performed after the fetching, if needed.
    @pyqtSlot(object, object)
    def step_2(self, result, operator):
        if operator != self.fetcher:
            return # This was not the operator I was waiting for.

        # Unsubscribe from further events on dataset
        self.root_dataset.operation_finished.disconnect(self.step_2)

        if result[0].N > 1000:
            self.subsampler = RandomSampler()
            self.subsampler.set_input({
                'parent': result[0]
            })
            self.subsampler.set_parameters({
                'k': 1000,
                'save_support': False
            })
            result[0].operation_finished.connect(self.step_3)
            result[0].perform_operation(self.subsampler)
        else:
            self.subsampler = None
            self.step_3(result, None)

    # Embed operation
    @pyqtSlot(object, object)
    def step_3(self, result, operator):
        if operator != self.subsampler:
            return # This was not the operator I was waiting for.
        if operator is not None:
            # Unsubscribe from further events on dataset
            result[0].parent().operation_finished.disconnect(self.step_3)

        parent = result[0]
        representatives_nd = self.query_nd
        representatives_2d = self.query_2d

        self.embedder = LAMPEmbedder()
        self.embedder.set_input({
            'parent': parent,
            'representatives_nd': representatives_nd,
            'representatives_2d': representatives_2d
        })
        self.embedder.set_parameters({
            'n_hidden_features': parent.hidden_features()
        })

        parent.operation_finished.connect(self.done)
        parent.perform_operation(self.embedder)

    # Embed operation
    @pyqtSlot(object, object)
    def done(self, result, operator):
        if operator != self.embedder:
            return # This was not the operator I was waiting for.
        print('Hierarchical zoom done! Emitting result.')
        self.when_done.emit(result[0])

    @classmethod
    def input_description(cls):
        return {
            'parent': (Dataset, False),
            'selected': (Dataset, False)
        }

    def input(self):
        return self._input

    def set_input(self, inputs):
        self._input = inputs

    @classmethod
    def parameters_description(cls):
        return {
            'center': (np.ndarray, None),
            'radius': (np.float, None),
            'x_dim': (int, 0),
            'y_dim': (int, 0)
        }

    def parameters(self):
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters


