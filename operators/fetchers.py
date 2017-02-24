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

from functools import reduce

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

class KNNFetcher(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        nd_dataset = self.input()['nd_dataset']
        query_nd = self.input()['query_nd']
        query_2d = self.input()['query_2d']

        N_max = self.parameters()['N_max']
        N_fetch = N_max - query_nd.N

        # Hide the hidden features
        X_use, X_hidden = Operator.hide_features(nd_dataset.data(), nd_dataset.hidden_features())
        X_query_use, X_query_hidden = Operator.hide_features(query_nd.data(), query_nd.hidden_features())
        Y_query_use, Y_query_hidden = Operator.hide_features(query_2d.data(), query_2d.hidden_features())

        # Find the point in the 2d selection closest to the center (cursor)
        nn = NearestNeighbors(n_neighbors=N_fetch) # Can probably query fewer points..
        nn.fit(X_use)
        distss, idcss = nn.kneighbors(X_query_use, return_distance=True)

        def red_func(tup1, tup2):
            dists1, idcs1 = tup1
            dists2, idcs2 = tup2

            idcs1_alone = np.setdiff1d(idcs1, idcs2, assume_unique=True)
            idcs2_alone = np.setdiff1d(idcs2, idcs1, assume_unique=True)
            idcs_intersect = np.intersect1d(idcs1, idcs2, assume_unique=True)
            
            try:
                idx1 = np.hstack([np.where(idcs1 == idx)[0] for idx in idcs1_alone])
            except IndexError:
                idx1 = []
            dists1_alone = dists1[idx1]
            try:
                idx2 = np.hstack([np.where(idcs2 == idx)[0] for idx in idcs2_alone])
            except IndexError:
                idx2 = []
            dists2_alone = dists2[idx2]

            idx1 = np.hstack([np.where(idcs1 == idx)[0] for idx in idcs_intersect])
            idx2 = np.hstack([np.where(idcs2 == idx)[0] for idx in idcs_intersect])
            dists_intersect = np.minimum(dists1[idx1], dists2[idx2])


            dists = np.hstack([dists1_alone, dists2_alone, dists_intersect])
            idcs = np.hstack([idcs1_alone, idcs2_alone, idcs_intersect])

            return dists, idcs

        reduced_dists, reduced_idcss = reduce(
            red_func,
            zip(distss, idcss)
        )

        print(len(reduced_idcss))

        out_dataset = Selection('F({}, {})'.format(query_nd.name(), nd_dataset.name()), nd_dataset, idcs=X_radius_neighbors, hidden=nd_dataset.hidden_features())
        self.set_output(out_dataset)