from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from modules.lense import Lense
from modules.dataset import Dataset, Selection, Embedding
from modules.operator import Operator

class LenseSelector(Operator):

    def __init__(self):
        super().__init__()

    def set_input(self, inputs):
        self._input = inputs

    def run(self):
        in_dataset = self.input()['embedding']
        parent = self.input()['parent']

        X = in_dataset.X

        x = X[:, self.parameters()['x_dim']]
        y = X[:, self.parameters()['y_dim']]
        Y = np.column_stack([x, y])

        p = self.parameters()['lense'].world_coordinates()
        p = np.array([[p.x(), p.y()]]) # Convert to np array
        
        radius = self.parameters()['lense'].world_radius()

        idcs = np.linalg.norm(Y - p, axis=1) <= radius
        
        X_filtered = parent.X[idcs, :]

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