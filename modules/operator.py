from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

from modules.dataset import Dataset


class Operator(QThread):

    def __init__(self):
        super().__init__()

    def input(self):
        return self._input

    def set_input(self, dataset):
        self._input = (dataset,)

    def set_inputs(self, datasets):
        self._input = tuple(datasets)

    def output(self):
        return self._output

    def set_output(self, dataset):
        self._output = (dataset,)

    def set_outputs(self, datasets):
        self._output = tuple(datasets)

    def parameters(self):
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters

    @abc.abstractmethod
    def run(self):
        """Method that should run the operator"""

    @classmethod
    @abc.abstractmethod
    def parameters_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """

    @classmethod
    @abc.abstractmethod
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
