from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from model.dataset import Dataset


class Operator(QThread, QObject):

    has_results = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.finished.connect(self.notify_caller)

    def input(self):
        return self._input

    def set_input(self, inputs):
        self._input = inputs

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

    def notify_caller(self):
        self.has_results.emit(self)

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

    @staticmethod
    def hide_features(X, n_hidden_features):
        n_features = X.shape[1] - n_hidden_features

        # Filter out the subset of features that can be used.
        X_use = X[:, :n_features]

        # Filter out the subset that cannot be used (because hidden!).
        X_hidden = X[:, -n_hidden_features:]
        
        return X_use, X_hidden
