import numpy as np
import numpy.ma as ma
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

from model import Dataset
from operators.root_selection import root_selection
from operators.random_sampling import random_sampling

# Return a dataset that contains the n_samples closest points in the root dataset,
# closest to the root observations corresponding to the samples in query.
def knn_fetching_naive_zi(query, n_samples, remove_query_points=True, sort=True, verbose=True):
    t_0 = time.time()

    X = Dataset.root.data()
    query_idcs = query.indices()

    # Query data
    Y = X[query_idcs, :]

    # Retrieve the root data on indices where the query is NOT.
    # (I.e., all candidates)
    source_idcs = np.delete(np.arange(Dataset.root.n_points()), query_idcs)
    X_search = X[source_idcs, :]

    # Compute smallest distances from all root points to all query points.
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
        print('knn_fetching_naive took {:.2f} seconds.'.format(time.time() - t_0))

    return dataset

# Return a dataset that contains the n_samples closest points in the root dataset,
# closest to the root observations corresponding to the samples in query.
def knn_fetching_zi(query_nd, n_samples, remove_query_points=True, sort=True, verbose=1):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())
    tree = Dataset.root_tree()

    t_0 = time.time()

    # In the worst case, we need to fetch this many points per query point.
    # (Because everything can overlap)
    # k = n_samples + query_nd.n_points()
    k = n_samples

    debug_time = time.time()
    dists, indices = tree.query(query_nd.data(), k=k)
    if verbose > 1:
        print('\tQuerying tree took {:.2f} s'.format(time.time() - debug_time))

    # dists: Every row 'i' contains the k smallest distances from query point 'i' to all root data points.
    # indices: Every row contains indices to root data, corresponding to the values in 'dists'.

    # Mask array that masks indices that are also in the query.
    # (We don't want these as result.)
    mask = np.zeros_like(indices, dtype=np.bool)
    if remove_query_points:
        debug_time = time.time()
        for i in range(indices.shape[0]):
            mask[i, :] = np.in1d(indices[i, :], query_nd.indices())
        if verbose > 1:
            print('\tRemoving query points took {:.2f} s'.format(time.time() - debug_time))

    m_indices = ma.masked_array(indices, mask=mask)
    m_dists = ma.masked_array(dists, mask=mask)
    indices_c = m_indices.compressed()
    dists_c = m_dists.compressed()

    debug_time = time.time()
    idx_sort = np.argsort(indices_c)
    indices_c = indices_c[idx_sort]
    dists_c = dists_c[idx_sort]
    if verbose > 1:
        print('\tSorting indices took {:.2f} s'.format(time.time() - debug_time))

    # Get the unique indices, and where they are in the array
    unique_idcs, idx_starts, counts = np.unique(indices_c, return_index=True, return_counts=True)

    # Reduce to the smallest distance per unique index.
    debug_time = time.time()
    min_dists = np.zeros_like(unique_idcs)
    for i, (idx_start, count) in enumerate(zip(idx_starts, counts)):
        min_dists[i] = dists_c[idx_start:idx_start + count].min()
    if verbose > 1:
        print('\tFinding min_dists took {:.2f} s'.format(time.time() - debug_time))

    if unique_idcs.size > n_samples:
        # Use only the n_samples closest samples for the result.
        closest_idcs = np.argpartition(min_dists, n_samples)[:n_samples]
        idcs_in_root = unique_idcs[closest_idcs]
    else:
        idcs_in_root = unique_idcs
        if unique_idcs.size < n_samples:
            print('WARNING: Returning fewer samples than requested, because more were not found!')

    if sort:
        idcs_in_root.sort()

    data = Dataset.root.data()[idcs_in_root, :]
    dataset = Dataset(data, idcs_in_root, name='KNN fetching') 

    if verbose:
        print('knn_fetching took {:.2f} seconds.\n'.format(time.time() - t_0))

    return dataset


def knn_fetching_zo(query_nd, k, n_samples, sort=True, verbose=2):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())
    tree = Dataset.root_tree()

    t_0 = time.time()

    debug_time = time.time()
    _, indices = tree.query(query_nd.data(), k=k)
    if verbose > 1:
        print('\tQuerying tree took {:.2f} s'.format(time.time() - debug_time))

    # Get the unique indices, and where they are in the array
    unique_idcs = np.unique(indices.flatten())
    
    if sort:
        unique_idcs.sort()

    query_result_data = Dataset.root.data()[unique_idcs, :]
    query_result = Dataset(query_result_data, unique_idcs, name='Query result.')

    if verbose > 1:
        print('\tFound {} unique observations for zoom-out.'.format(unique_idcs.size))

    if unique_idcs.size > n_samples:
        if verbose > 1:
            print('\tSubsampling {} observations to {}.'.format(unique_idcs.size, n_samples))
        dataset = random_sampling(query_result, n_samples)
    else:
        dataset = query_result
    


    if verbose:
        print('knn_fetching_zo took {:.2f} seconds.\n'.format(time.time() - t_0))

    return dataset