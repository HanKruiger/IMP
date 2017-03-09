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
            'radius': (np.float, None)
        }

class KNNFetcher(Operator):

    def __init__(self):
        super().__init__()

    def run(self):
        print('KNNFetcher:')
        nd_dataset = self.input()['nd_dataset']
        query_nd = self.input()['query_nd']
        query_2d = self.input()['query_2d']

        N_max = self.parameters()['N_max']
        N_fetch = N_max - query_nd.n_points()

        # Hide the hidden features
        X_query_use, X_query_hidden = Operator.hide_features(query_nd.data(), query_nd.hidden_features())
        Y_query_use, Y_query_hidden = Operator.hide_features(query_2d.data(), query_2d.hidden_features())

        # Remove points in X_query_use from X_use, because we don't want the same points as the query.
        dataset_no_query_idcs = np.delete(np.arange(nd_dataset.n_points()), query_nd.indices(), axis=0)
        nd_dataset_noquery = Selection('S_m({})'.format(nd_dataset.name), nd_dataset, idcs=dataset_no_query_idcs, hidden=nd_dataset.hidden_features())
        
        X_use, X_hidden = Operator.hide_features(nd_dataset_noquery.data(), nd_dataset_noquery.hidden_features())
        print(X_use.shape)

        # Find the point in the 2d selection closest to the center (cursor)
        nn = NearestNeighbors(n_neighbors=N_fetch) # Can probably query fewer points..
        nn.fit(X_use)
        nbr_dists, nbr_idcs = nn.kneighbors(X_query_use, return_distance=True)

        print('\tFetched {} neighbours.'.format(nbr_idcs.size))

        U_idcs = np.unique(nbr_idcs)
        U_dists = np.full(U_idcs.size, np.inf)

        print('\tReducing neighbours into {} ({:.2f}%) unique neighbours.'.format(U_idcs.size, 100 * U_idcs.size / nbr_idcs.size))

        # Dict that maps indices in X_use to indices in U_idcs/U_dists
        idx_to_idx = dict([(i, j) for j, i in enumerate(U_idcs)])

        for i in range(nbr_idcs.shape[0]):
            for j in range(nbr_idcs.shape[1]):
                k = nbr_idcs[i, j] # Index in X_use
                dist = nbr_dists[i, j] # New distance
                l = idx_to_idx[k] # Index in U_idcs
                U_dists[l] = min(U_dists[l], dist) # Take minimum of the two

        print('\tMinimum distance: {}'.format(U_dists.min()))
        print('\tMaximum distance: {}'.format(U_dists.max()))
        print('\tsize(U_dists): {}'.format(U_dists.size))
        print('\tN_fetch: {}'.format(N_fetch))


        if U_dists.size == N_fetch:
            X_knn = U_idcs
        else:
            closest_nbrs = np.argpartition(U_dists, N_fetch)[:N_fetch]
            X_knn = U_idcs[closest_nbrs] # Indices the data in nd_dataset_noquery
        
        X_knn = nd_dataset_noquery.indices()[X_knn] # Indices the data in nd_dataset
        print('\tUsing {} closest neighbours.'.format(X_knn.size))

        X_knn = np.concatenate([X_knn, query_nd.indices()])

        out_dataset = Selection('F({}, {})'.format(query_nd.name(), nd_dataset.name()), nd_dataset, idcs=X_knn, hidden=nd_dataset.hidden_features())
        self.set_output(out_dataset)