from model import *
import numpy as np
from scipy.sparse.linalg import svds
from numpy.linalg import svd


class Sampling(Selection):

    def __init__(self, parent, n_samples, name=None, hidden=None):
        self._n_points = n_samples
        self._n_dimensions = parent.n_dimensions()

        super().__init__(parent, idcs=None, name=name, hidden=hidden)


class RandomSampling(Sampling):

    def __init__(self, parent, n_samples, name=None, hidden=None):
        if name is None:
            name = 'Rnd({})'.format(parent.name())

        super().__init__(parent, n_samples, name=name, hidden=hidden)

        sampler = RandomSampling.RandomSamplingWorker(parent, n_samples)
        self.spawn_thread(sampler, self.set_indices_in_parent, waitfor=(parent,))

    class RandomSamplingWorker(Dataset.Worker):

        def __init__(self, parent, n_samples):
            super().__init__()
            self.parent = parent
            self.n_samples = n_samples

        def work(self):
            idcs_in_parent = np.random.choice(self.parent.n_points(), self.n_samples, replace=False)
            idcs_in_parent.sort()
            self.ready.emit(idcs_in_parent)


class SVDBasedSampling(Sampling):

    def __init__(self, parent, n_samples, name=None, hidden=None):
        if name is None:
            name = 'Rnd({})'.format(parent.name())

        super().__init__(parent, n_samples, name=name, hidden=hidden)

        sampler = SVDBasedSampling.SVDBasedSampler(parent, n_samples)
        self.spawn_thread(sampler, self.set_indices_in_parent, waitfor=(parent,))

    class SVDBasedSampler(Dataset.Worker):

        def __init__(self, parent, n_samples, k=None):
            super().__init__()
            self.parent = parent
            self.n_samples = n_samples
            self.k = k

        def work(self):
            X = self.parent.data()
            c = self.n_samples

            if self.k is None:
                k = min(X.shape) // 2 + 1  # As suggested by Joia et al.

            if k < X.shape[1]:
                _, _, V_T = svds(X.T, k=k)
            else:
                _, _, V_T = svd(X.T, full_matrices=False)

            pi = (V_T ** 2).sum(axis=0)

            # Get the c indices with the c largest value in pi (in no particular order)
            idcs_in_parent = np.argpartition(pi, -c)[-c:]

            self.ready.emit(idcs_in_parent)
