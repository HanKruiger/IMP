import numpy as np

from operators.operator import Operator
from model.dataset import Selection, Dataset
import abc
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

# Return a dataset that contains the k 
def knn_fetch(query_nd, nd_dataset, k):
    print('KNNFetch, k = {}:'.format(k))

    # Hide the hidden features
    X_query_use, X_query_hidden = Operator.hide_features(query_nd.data(), query_nd.hidden_features())

    # Remove points in X_query_use from X_use, because we don't want the same points as the query.
    dataset_no_query_idcs = np.delete(np.arange(nd_dataset.n_points()), query_nd.indices(), axis=0)
    nd_dataset_noquery = Selection('S_m({})'.format(nd_dataset.name), nd_dataset, idcs=dataset_no_query_idcs, hidden=nd_dataset.hidden_features())
    
    X_use, X_hidden = Operator.hide_features(nd_dataset_noquery.data(), nd_dataset_noquery.hidden_features())
    print(X_use.shape)

    # Find the point in the 2d selection closest to the center (cursor)
    nn = NearestNeighbors(n_neighbors=k) # Can probably query fewer points..
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
    print('\tk: {}'.format(k))


    if U_dists.size == k:
        X_knn = U_idcs
    else:
        closest_nbrs = np.argpartition(U_dists, k)[:k]
        X_knn = U_idcs[closest_nbrs] # Indices the data in nd_dataset_noquery
    
    X_knn = nd_dataset_noquery.indices()[X_knn] # Indices the data in nd_dataset
    print('\tUsing {} closest neighbours.'.format(X_knn.size))

    X_knn = np.concatenate([X_knn, query_nd.indices()])

    out_dataset = Selection('F({}, {})'.format(query_nd.name(), nd_dataset.name()), nd_dataset, idcs=X_knn, hidden=nd_dataset.hidden_features())
    return out_dataset