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
        Y = np.column_stack([dataset.X for dataset in self.input()])
        self.set_output(Dataset(self.input()[0].name, parent=self.input()[0], relation='hcat', X=Y))

    @classmethod
    def parameters_description(cls):
        return []

    @classmethod
    def input_description(cls):
        return [
            ('dataset1', Dataset),
            ('dataset2', Dataset)
        ]
