import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

from model import Dataset

# Return a dataset that contains the k closest points in X closest to query_idcs (but not also in query_idcs!).
def knn_fetching(source, query, n_samples, sort=True, verbose=True):
    t_0 = time.time()

    X = source.data()
    query_idcs = query.indices()

    # Query data
    Y = X[query_idcs, :]

    # Indices the source data on indices where the query is NOT.
    # (I.e., all candidates)
    source_idcs = np.delete(np.arange(source.n_points()), query_idcs)

    # Holds indexes in X, and the distance from those observations to any query point.
    closest = dict()

    # Holds the distance to the furthest closest observation.
    max_dist = -np.inf

    for i in source_idcs:
        # Compute the smallest distance from the candidate to any query sample.
        dist = np.min(np.linalg.norm(Y - X[i, :], axis=1))

        if len(closest) < n_samples:
            # Still room for more candidates, add in any case.
            closest[i] = dist
            max_dist = max(max_dist, dist)
        elif dist < max_dist:
            # Throw the furthest candidate out
            j = max(closest.keys(), key=lambda idx: closest[idx])
            del closest[j]

            # Add this candidate, because it's at least closer than j.
            closest[i] = dist

            # Recompute it.
            max_dist = max(closest.values())

    idcs_in_root = np.array(list(closest.keys()))
    if sort:
        idcs_in_root.sort()
    data = X[idcs_in_root, :]
    dataset = Dataset(data, idcs_in_root, name='KNN fetching')

    if verbose:
        print('knn_fetching took {:.2f} seconds.'.format(time.time() - t_0))

    return dataset