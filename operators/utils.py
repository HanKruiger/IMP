import numpy as np

import abc
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist


def hide_features(X, n_hidden_features):
    if n_hidden_features == 0:
        return X, np.empty(shape=(X.shape[0], 0))
    n_features = X.shape[1] - n_hidden_features

    # Filter out the subset of features that can be used.
    X_use = X[:, :n_features]

    # Filter out the subset that cannot be used (because hidden!).
    X_hidden = X[:, -n_hidden_features:]

    return X_use, X_hidden


# Return a dataset that contains the k closest points in nd_dataset closest to query_nd (but not also in query_nd!).
def knn_fetch(X, query_idcs, k):
    X_query = X[query_idcs, :]

    # Remove points in X_query_use from X_use, because we don't want the same points as the query.
    no_query_idcs = np.delete(np.arange(X.shape[0]), query_idcs, axis=0)
    X_no_query = X[no_query_idcs, :]

    # Find the point in the 2d selection closest to the center (cursor)
    nn = NearestNeighbors(n_neighbors=k)  # Can probably query fewer points..
    nn.fit(X_no_query)
    nbr_dists, nbr_idcs = nn.kneighbors(X_query, return_distance=True)

    U_idcs = np.unique(nbr_idcs)
    U_dists = np.full(U_idcs.size, np.inf)

    # Dict that maps indices in X_use to indices in U_idcs/U_dists
    idx_to_idx = dict([(i, j) for j, i in enumerate(U_idcs)])

    for i in range(nbr_idcs.shape[0]):
        for j in range(nbr_idcs.shape[1]):
            l = nbr_idcs[i, j]  # Index in X_use
            dist = nbr_dists[i, j]  # New distance
            m = idx_to_idx[l]  # Index in U_idcs
            U_dists[m] = min(U_dists[m], dist)  # Take minimum of the two

    if U_dists.size <= k:
        X_knn = U_idcs
    else:
        closest_nbrs = np.argpartition(U_dists, k)[:k]
        X_knn = U_idcs[closest_nbrs]  # Indices the data in nd_dataset_noquery

    X_knn = no_query_idcs[X_knn]  # Indices the data in nd_dataset

    # X_knn = np.concatenate([X_knn, query_nd.indices_in_root()])
    return np.sort(X_knn)
