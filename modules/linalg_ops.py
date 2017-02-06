from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from modules.dataset import Dataset
from modules.operator import Operator

class HorizontalConcat(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        in_dataset_1 = self.input()[0][0]
        in_dataset_2 = self.input()[1][0]

        assert(in_dataset_1.N == in_dataset_2.N)
        
        Y = np.column_stack([in_dataset_1.X, in_dataset_2.X])
        
        out_dataset = Dataset(in_dataset_1.name + '_cat', parent=in_dataset_1, relation='hcat', X=Y)
        self.set_output(out_dataset)

    @classmethod
    def parameters_description(cls):
        return []

    @classmethod
    def input_description(cls):
        return [
            ('dataset1', Dataset, False),
            ('dataset2', Dataset, False)
        ]
