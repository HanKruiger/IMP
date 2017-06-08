import numpy as np
import numpy.ma as ma
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

from model import Dataset
from operators.root_selection import root_selection


def knn_fetching_zo_1(query_nd, k, N_max, sort=True, verbose=2):
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
        dataset = query_result.random_sampling(N_max)
    else:
        dataset = query_result

    if verbose:
        print('knn_fetching_zo took {:.2f} seconds.\n'.format(time.time() - t_0))

    return dataset


def knn_fetching_zo_2(query_nd, N_max, zoom_factor=1.1, sort=True, verbose=2, tolerance=0.1, max_iters=100):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())

    t_0 = time.time()

    sphere_query = query_nd.bounding_hypersphere(smooth=True)

    L_lower = N_max
    L_upper = None
    L_candidate = L_lower

    # Binary search
    iters = 0
    while True and iters < max_iters:
        D_s = Dataset.root.random_sampling(L_candidate)
        # result = pointset_knn_naive(query_nd, N_max, source=D_s)
        result = D_s.knn_pointset(N_max, query_dataset=query_nd, remove_query_points=False, method='bruteforce')

        sphere_result = result.bounding_hypersphere(smooth=True)
        normalized_error = sphere_result.radius() / (zoom_factor * sphere_query.radius()) - 1

        if verbose > 1:
            print('L search:\n\tcandidate: {}\n\tlower: {}\n\tupper: {}\n\terror: {}\n\tHS_q in HS_r: {}'.format(L_candidate, L_lower, L_upper, normalized_error, sphere_query in sphere_result))

        # Note: We only accept positive errors, because we want to guarantee radius increase.
        if normalized_error > 0 and normalized_error < tolerance:
            # We're within tolerance!
            break
        elif L_lower == L_upper and normalized_error > 0:
            # End of search. Probably not optimal.
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

        # Determine new candidate for next search iteration.
        if L_upper is None:
            L_candidate = 2 * L_lower
        else:
            L_candidate = round((L_lower + L_upper) / 2)

        iters += 1

    if verbose:
        print('knn_fetching_zo_2 took {:.2f} seconds. ({} search iterations)\n'.format(time.time() - t_0, iters))

    if verbose and iters == max_iters:
        print('Warning: Reached max_iters (= {}) in binary search.'.format(max_iters))
    if verbose and L_lower == L_upper and normalized_error < tolerance:
        print('Warning: End of binary search. No optimal results for zoom out.')

    return result

def knn_fetching_zo_3(query_nd, N_max, zoom_factor=1.3, sort=True, verbose=2, tolerance=0.05, max_iters=100):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())

    t_0 = time.time()

    a_star = zoom_factor
    sphere_query = query_nd.bounding_hypersphere(smooth=True)

    L_lower = N_max
    L_upper = None
    L_candidate = L_lower

    # Binary search
    iters = 0
    while True and iters < max_iters:
        D_s = Dataset.root.random_sampling(L_candidate)
        # result = pointset_knn_naive(query_nd, N_max, source=D_s)
        result = D_s.knn_pointset(N_max, query_dataset=query_nd, remove_query_points=False, method='bruteforce')

        points_in_sphere = sphere_query.contains(result.data()).sum()
        a = result.n_points() / points_in_sphere

        error = a - a_star

        if verbose > 1:
            print('L search:\n\tcandidate: {}\n\tlower: {}\n\tupper: {}\n\terror: {}'.format(L_candidate, L_lower, L_upper, error))

        # Positive error: a is too large, so we need fewer points outside the sphere --> Make sampling denser
        # Negative error: a is too small, so we need more points outside the sphere --> Make sampling sparser

        # Note: We only accept positive errors, because we want to guarantee radius increase.
        if error > 0 and error < tolerance:
            # We're within tolerance!
            break
        elif L_lower == L_upper and error > 0:
            # End of search. Probably not optimal.
            break
        elif error < 0:
            # Sampling was too dense when using candidate.

            # If upper value had not been set yet, it means that we should return the sparsest sampling!
            # (Resulting in the current result, so break!)
            if L_upper is None and L_lower == N_max:
                break

            # Otherwise, update the upper bound
            L_upper = L_candidate
        elif error > 0:
            # Sampling was too sparse when using candidate.
            # Update the lower bound.
            L_lower = L_candidate

        # Determine new candidate for next search iteration.
        if L_upper is None:
            L_candidate = 2 * L_lower
        else:
            L_candidate = round((L_lower + L_upper) / 2)

        iters += 1

    if verbose:
        print('knn_fetching_zo_2 took {:.2f} seconds. ({} search iterations)\n'.format(time.time() - t_0, iters))

    if verbose and iters == max_iters:
        print('Warning: Reached max_iters (= {}) in binary search.'.format(max_iters))
    if verbose and L_lower == L_upper and error < tolerance:
        print('Warning: End of binary search. No optimal results for zoom out.')

    return result

# TODO: Make this compatible with k > 1
def knn_fetching_zo_4(query_nd, N_max, M, new_fraction=0.5, k=1, sort=True, verbose=2):
    assert(Dataset.root.n_dimensions() == query_nd.n_dimensions())

    t_0 = time.time()

    sphere = query_nd.bounding_hypersphere(smooth=False)

    # Number of points from query_nd we want to keep.
    N_keep = round(N_max * (1 - new_fraction))

    query_nd_subsampled = query_nd.random_sampling(N_keep)

    # Number of points we need to add.
    N_fetch = N_max - N_keep

    fetched_idcs = np.zeros(N_fetch, dtype=np.int)
    for i in range(N_fetch):
        new_idx = None
        while new_idx is None or new_idx in fetched_idcs[:i]:
            D_s = Dataset.root.random_sampling(M)
            D_s_outside_sphere = D_s.select_logical(~sphere.contains(D_s.data()))
            # Just to be sure, remove any points from query_nd_subsampled.
            D_s_outside_sphere = D_s_outside_sphere.select_logical(~np.in1d(D_s_outside_sphere.indices(), query_nd_subsampled.indices()))
            singleton_dataset = D_s_outside_sphere.knn_pointset(k, query_dataset=query_nd, method='bruteforce', verbose=False)
            new_idx = singleton_dataset.indices()[0]

        print('{}/{}'.format(i, N_fetch))
        fetched_idcs[i] = new_idx
    
    fetched_data = Dataset.root.data()[fetched_idcs, :] # Is it faster to also fill this in the loop?
    fetched_dataset = Dataset(fetched_data, fetched_idcs)

    # Take union of the subsampled query and the newly fetched dataset.
    result = query_nd_subsampled + fetched_dataset

    return result
