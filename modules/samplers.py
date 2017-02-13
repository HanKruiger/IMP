from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from modules.dataset import Dataset, Sampling
from modules.operator import Operator

class RandomSampler(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        in_dataset = self.input()['parent']

        X = in_dataset.X

        N = X.shape[0]
        k = self.parameters()['k']
        idcs = np.random.choice(N, k, replace=False)
        Y = X[idcs, :]
        
        out_dataset = Sampling(in_dataset.name() + 's', parent=in_dataset, X=Y, hidden=in_dataset.hidden_features())
        self.set_output(out_dataset)

    @classmethod
    def parameters_description(cls):
        return {
            'k': (int, 1000),
        }

    @classmethod
    def input_description(cls):
        return {
            'parent': Dataset
        }
