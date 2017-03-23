from model import *

import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNFetching(Selection):

    def __init__(self, query_2d, n_points, name=None, hidden=None):
        root = query_2d.root()
        if name is None:
            name = 'F({}, {})'.format(root.name(), query_2d.name())
        if hidden is None:
            hidden = root.hidden_features()
        self._n_points = n_points
        super().__init__(root, idcs=None, name=name, hidden=hidden)

        query_nd = RootSelection(query_2d)
        fetcher = KNNFetching.KNNFetcher(query_nd, root, n_points)

        self.spawn_thread(fetcher, self.set_indices_in_parent, waitfor=(query_nd, root))

    class KNNFetcher(Dataset.Worker):

        def __init__(self, query_nd, root, n_points):
            super().__init__()
            self.query_nd = query_nd
            self.root = root
            self.n_points = n_points

        def work(self):
            X, _ = self.root.data(split_hidden=True)
            query_idcs = self.query_nd.indices_in_root()
            idcs_in_parent = self.knn_fetch(X, query_idcs, self.n_points)
            self.ready.emit(idcs_in_parent)

        # Return a dataset that contains the k closest points in X closest to query_idcs (but not also in query_idcs!).
        def knn_fetch(self, X, query_idcs, k):
            X_query = X[query_idcs, :]

            # Remove points in X_query_use from X_use, because we don't want the same points as the query.
            no_query_idcs = np.delete(np.arange(X.shape[0]), query_idcs, axis=0)
            X_no_query = X[no_query_idcs, :]

            k = min(k, X_no_query.shape[0] - 1)

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

            return np.sort(X_knn)
