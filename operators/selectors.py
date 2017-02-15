from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from widgets.gl_entities.lense import Lense
from model.dataset import Dataset, Selection, Embedding
from operators.operator import Operator

class LenseSelector(Operator):

    def __init__(self):
        super().__init__()

    def set_input(self, inputs):
        self._input = inputs

    def run(self):
        in_dataset = self.input()['embedding']
        parent = self.input()['parent']

        X = in_dataset.data()

        x = X[:, self.parameters()['x_dim']]
        y = X[:, self.parameters()['y_dim']]
        Y = np.column_stack([x, y])

        p = self.parameters()['lense'].world_coordinates()
        p = np.array([[p.x(), p.y()]]) # Convert to np array
        
        radius = self.parameters()['lense'].world_radius()

        idcs = np.linalg.norm(Y - p, axis=1) <= radius
        
        X_filtered = parent.data()[idcs, :]

        out_dataset = Selection(parent.name() + 's', parent=parent, X=X_filtered, hidden=parent.hidden_features())
        self.set_output(out_dataset)

    @classmethod
    def parameters_description(cls):
        return {
            'lense': (Lense, None),
            'x_dim': (int, 0),
            'y_dim': (int, 0)
        }

    @classmethod
    def input_description(cls):
        return {
            'parent': Dataset,
            'embedding': Dataset,
        }