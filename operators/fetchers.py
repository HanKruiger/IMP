from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model.dataset import Dataset, Selection
from operators.operator import Operator

import abc
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

class HypersphereFetcher(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        nd_dataset = self.input()['nd_dataset']
        query_nd = self.input()['query_nd']
        query_2d = self.input()['query_2d']

        center = self.parameters()['center']
        radius = self.parameters()['radius']

        # Hide the hidden features
        X_use, X_hidden = Operator.hide_features(nd_dataset.data(), nd_dataset.hidden_features())
        X_query_use, X_query_hidden = Operator.hide_features(query_nd.data(), query_nd.hidden_features())
        Y_query_use, Y_query_hidden = Operator.hide_features(query_2d.data(), query_2d.hidden_features())

        # Find the 'conversion factor' for distances from 2d to nd.
        lbd = pdist(X_query_use).sum() / pdist(Y_query_use).sum()

        # Find the point in the 2d selection closest to the center (cursor)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(Y_query_use)
        nearest_idx = nn.kneighbors(center.reshape(1, -1), return_distance=False).flatten()[0]
        # y_nearest = Y_query_use[nearest_idx, :] Not even needed.

        # Get the point in nd corresponding to that nearest point.
        x_nearest = X_query_use[nearest_idx, :]

        # Find the nd points inside the ball centered at x_nearest, with radius converted to nD with the factor.
        nn = NearestNeighbors(radius=radius * lbd)
        nn.fit(X_use)
        X_radius_neighbors = nn.radius_neighbors(x_nearest.reshape(1, -1), return_distance=False).flatten()[0]

        out_dataset = Selection('F({}, {})'.format(query_nd.name(), nd_dataset.name()), nd_dataset, idcs=X_radius_neighbors, hidden=nd_dataset.hidden_features())
        self.set_output(out_dataset)


    @classmethod
    def input_description(cls):
        """Method that should run parameters needed for the operator,
        along with their types and default values. """
        return {
            'nd_dataset': Dataset,
            'query_nd': Dataset,
            'query_2d': Dataset
        }

    @classmethod
    def parameters_description(cls):
        return {
            'center': (np.ndarray, None),
            'radius': (np.float, None),
            'k': (int, None)
        }
