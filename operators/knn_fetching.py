import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

from model import Dataset

# Return a dataset that contains the n_samples closest points in the root dataset,
# closest to the root observations corresponding to the samples in query.
def knn_fetching(query, n_samples, sort=True, verbose=True):
    t_0 = time.time()

    X = Dataset.root.data()
    query_idcs = query.indices()

    # Query data
    Y = X[query_idcs, :]

    # Retrieve the root data on indices where the query is NOT.
    # (I.e., all candidates)
    source_idcs = np.delete(np.arange(Dataset.root.n_points()), query_idcs)
    X_search = X[source_idcs, :]

    # Compute smallest distances from all root points to any query point.
    dists = cdist(Y, X_search, metric='euclidean').min(axis=0)
    # Retrieve indices (in X_search!) where the distances are smallest
    smallest_dist_idcs = np.argpartition(dists, n_samples)[:n_samples]
    # Get the corresponding indices in the root dataset
    idcs_in_root = source_idcs[smallest_dist_idcs]

    if sort:
        idcs_in_root.sort()

    data = X[idcs_in_root, :]
    dataset = Dataset(data, idcs_in_root, name='KNN fetching')

    if verbose:
        print('knn_fetching took {:.2f} seconds.'.format(time.time() - t_0))

    return dataset