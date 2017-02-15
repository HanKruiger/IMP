from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from model.dataset import Dataset, Merging
from operators.operator import Operator
from operators.selectors import LenseSelector
from operators.i_projectors import InverseProjector

class HierarchicalZoom(QObject):

    def initialize(self):
        self.selector = LenseSelector()
        self.selector.set_input({
            'embedding': self.input()['selected'],
            'parent': self.input()['parent']
        })
        self.selector.set_parameters({
            'center': self.parameters()['center'],
            'radius': self.parameters()['radius'],
            'x_dim': self.parameters()['x_dim'],
            'y_dim': self.parameters()['y_dim']
        })

        self.input()['parent'].operation_finished.connect(self.step_1)

    def run(self):
        self.input()['parent'].perform_operation(self.selector)

    @pyqtSlot(object, object)
    def step_1(self, result, operator):
        if operator != self.selector:
            return # This was not the operator I was waiting for.
        
        query_nd = result[0]
        query_2d = result[1]

        # Unsubscribe from further events on dataset
        self.input()['parent'].operation_finished.disconnect(self.step_1)

        root_dataset = self.input()['parent']
        while root_dataset.parent() is not None and not isinstance(root_dataset, Merging):
            root_dataset = root_dataset.parent()
        print(root_dataset)

        self.inverse_projector = InverseProjector()
        self.inverse_projector.set_input({
            'nd_dataset': root_dataset,
            'query_nd': query_nd,
            'query_2d': query_2d
        })
        self.inverse_projector.set_parameters({
            'center': self.parameters()['center'],
            'radius': self.parameters()['radius']    
        })

        root_dataset.operation_finished.connect(self.step_2)
        root_dataset.perform_operation(self.inverse_projector)
        

    @pyqtSlot(object, object)
    def step_2(self, result, operator):
        if operator != self.inverse_projector:
            return # This was not the operator I was waiting for.

        # Unsubscribe from further events on dataset
        root_dataset.operation_finished.disconnect(self.step_2)

        print('In step 2')


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


