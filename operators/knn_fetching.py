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
def pointset_knn_naive(query, n_samples, source=None, remove_query_points=True, sort=True, verbose=True, return_distances=False):
    t_0 = time.time()

    if source is None:
        source = Dataset.root

    # Compute smallest distances from all root points to all query points.
    dists = cdist(query.data(), source.data(), metric='euclidean').min(axis=0)
    # Retrieve indices (in source!) where the distances are smallest
    if n_samples == dists.size:
        smallest_dist_idcs = np.arange(n_samples)
    else:
        smallest_dist_idcs = np.argpartition(dists, n_samples)[:n_samples]
        
    # Get the corresponding indices in the root dataset
    idcs_in_root = source.indices()[smallest_dist_idcs]

    if sort:
        order = np.argsort(idcs_in_root)
        idcs_in_root = idcs_in_root[order]
        smallest_dist_idcs = smallest_dist_idcs[order]

    data = source.data()[smallest_dist_idcs, :]
    dataset = Dataset(data, idcs_in_root, name='KNN fetching')

    if verbose:
        print('knn_fetching_naive took {:.2f} seconds.'.format(time.time() - t_0))

    if return_distances:
        return dataset, dists[smallest_dist_idcs]
    else:
        return dataset

# Return a dataset that contains the n_samples closest points in the root dataset,
# closest to the root observations corresponding to the samples in query.
def knn_fetching_zi(query_nd, n_samples, k, remove_query_points=True, sort=True, verbose=2):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())
    tree = Dataset.root.tree()

    t_0 = time.time()

    # In the worst case, we need to fetch this many points per query point.
    # (Because everything can overlap)
    # k = n_samples + query_nd.n_points()

    debug_time = time.time()
    # dists, indices = tree.query(query_nd.data(), k=k)
    dists = np.zeros((query_nd.n_points(), k))
    indices = np.zeros((query_nd.n_points(), k), dtype=np.int)
    for i, root_id in enumerate(query_nd.indices()):
        res = tree.get_nns_by_item(root_id, k, include_distances=True)
        indices[i, :] = res[0]
        dists[i, :] = res[1]
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

    if unique_idcs.size < n_samples:
        print('\t{}-nn was too few (only {} unique results, needed {}) retrying with {}-nn...'.format(k, unique_idcs.size, n_samples, 2*k))
        return knn_fetching_zi(query_nd, n_samples, k * 2, remove_query_points=remove_query_points, sort=sort, verbose=verbose)

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


def knn_fetching_zo(query_nd, k, N_max, sort=True, verbose=2):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())
    tree = Dataset.root.tree()

    t_0 = time.time()

    debug_time = time.time()

    indices = np.zeros((query_nd.n_points(), k), dtype=np.int)
    for i, root_id in enumerate(query_nd.indices()):
        res = tree.get_nns_by_item(root_id, k, include_distances=False)
        indices[i, :] = res

    if verbose > 1:
        print('\tQuerying tree took {:.2f} s'.format(time.time() - debug_time))

    # Get the unique indices, and where they are in the array
    unique_idcs = np.unique(indices.flatten())

    if verbose > 1:
        print('\tSearched for {} neighbours of {} observations.'.format(k, query_nd.n_points()))
        print('\tFound {} observations ({} unique)'.format(indices.size, unique_idcs.size))
    
    if sort:
        unique_idcs.sort()

    query_result_data = Dataset.root.data()[unique_idcs, :]
    query_result = Dataset(query_result_data, unique_idcs, name='Query result.')

    if verbose > 1:
        print('\tFound {} unique observations for zoom-out.'.format(unique_idcs.size))

    if unique_idcs.size > N_max:
        if verbose > 1:
            print('\tSubsampling {} observations to {}.'.format(unique_idcs.size, N_max))
        dataset = random_sampling(query_result, N_max)
    else:
        dataset = query_result

    if verbose:
        print('knn_fetching_zo took {:.2f} seconds.\n'.format(time.time() - t_0))

    return dataset

def knn_fetching_zo_2(query_nd, N_max, zoom_factor=1.2, sort=True, verbose=2, tolerance=0.1, max_iters=100):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())

    t_0 = time.time()

    L_lower = N_max
    L_upper = None
    L_candidate = None

    # Binary (?) search
    iters = 0
    while True and iters < max_iters:
        if L_upper is None:
            if L_candidate is None:
                L_candidate = L_lower
            else:
                L_candidate = L_lower * 2
        else:
            L_candidate = round((L_lower + L_upper) / 2)

        D_s = random_sampling(Dataset.root, L_candidate)
        result, dists = pointset_knn_naive(query_nd, N_max, source=D_s, return_distances=True)
        
        normalized_error = result.radius() / (zoom_factor * query_nd.radius()) - 1
        
        if verbose > 1:
            print('L search:\n\tcandidate: {}\n\tlower: {}\n\tupper: {}\n\terror: {}'.format(L_candidate, L_lower, L_upper, normalized_error))
        
        # Note: We do not check for absolute error, because we want to guarantee radius increase.
        if normalized_error > 0 and normalized_error < tolerance:
            # We're within tolerance!
            break
        elif L_lower == L_upper and normalized_error > 0:
            break
        elif normalized_error < 0:
            # Sampling was too dense when using candidate.

            # If upper value had not been set yet, it means that we should return the sparsest sampling!
            # (Resulting in the current result, so break!)
            if L_upper is None and L_lower == N_max:
                break

            # Otherwise, update the upper bound
            L_upper = L_candidate
        elif normalized_error > 0:
            # Sampling was too sparse when using candidate.
            # Update the lower bound.
            L_lower = L_candidate

        iters += 1

    if verbose:
        print('knn_fetching_zo_2 took {:.2f} seconds. ({} search iterations)\n'.format(time.time() - t_0, iters))

    if verbose and iters == max_iters:
        print('Warning: reached max_iters in binary search.')


    return result