from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from modules.dataset import Dataset


class Operator(QThread):

    def __init__(self):
        super().__init__()

    def input(self):
        return self._input

    def set_input(self, dataset, hidden_features):
        self._input = (tuple(dataset, hidden_features),)

    def set_inputs(self, datasets, hidden_featuress):
        self._input = tuple(zip(tuple(datasets), tuple(hidden_featuress)))

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

    @staticmethod
    def hide_features(X, hidden_features):
        n_features = X.shape[1] - len(hidden_features)

        mask = np.ones(X.shape[1], dtype=bool)
        mask[hidden_features,] = False
        features = np.arange(X.shape[1])[mask,]
        del mask
        assert(len(features) == n_features)

        # Filter out the subset of features that can be used.
        X_use = X[:, features]

        # Filter out the subset that cannot be used (because hidden!).
        X_hidden = X[:, hidden_features]
        
        return X_use, X_hidden
