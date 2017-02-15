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

    def run(self):
        in_dataset = self.input()['embedding']
        parent = self.input()['parent']

        X = in_dataset.data()

        x = X[:, self.parameters()['x_dim']]
        y = X[:, self.parameters()['y_dim']]
        Y = np.column_stack([x, y])

        p = self.parameters()['center']
        radius = self.parameters()['radius']

        idcs = np.linalg.norm(Y - p, axis=1) <= radius
        idcs = np.flatnonzero(idcs)

        out_dataset_nd = Selection('Se({})'.format(parent.name()), parent=parent, idcs=idcs, hidden=parent.hidden_features())
        out_dataset_2d = Selection('Se({})'.format(in_dataset.name()), parent=in_dataset, idcs=idcs, hidden=in_dataset.hidden_features())
        
        self.set_outputs([out_dataset_nd, out_dataset_2d])

    @classmethod
    def parameters_description(cls):
        return {
            'center': (np.ndarray, None),
            'radius': (np.float, None),
            'x_dim': (int, 0),
            'y_dim': (int, 0),
        }

    @classmethod
    def input_description(cls):
        return {
            'parent': Dataset,
            'embedding': Dataset,
        }