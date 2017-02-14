from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from modules.dataset import Dataset, Merging
from modules.operator import Operator

class HorizontalConcat(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        in_dataset_1 = self.input()['parent']
        in_dataset_2 = self.input()['source']

        assert(in_dataset_1.N == in_dataset_2.N)
        
        Y = np.column_stack([in_dataset_1.X, in_dataset_2.X])
        
        hidden = 0
        if self.parameters()['add_as_hidden']:
            hidden += in_dataset_2.m

        out_dataset = Merging(in_dataset_1.name(), in_dataset_1, Y, hidden=hidden)
        self.set_output(out_dataset)

    @classmethod
    def parameters_description(cls):
        return {
            'add_as_hidden': (bool, True)
        }

    @classmethod
    def input_description(cls):
        return {
            'parent': (Dataset, False),
            'source': (Dataset, False)
        }
