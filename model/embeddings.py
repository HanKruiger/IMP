from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import numpy as np


class Embedding(Dataset):

    def __init__(self, parent, n_dimensions, name=None, hidden=None):
        if name is None:
            name = 'E({})'.format(parent.name())

        super().__init__(name, parent, data=None, hidden=hidden)

        self._n_dimensions = n_dimensions + self.hidden_features()

    def root(self):
        return self.parent().root()

    def indices_in_parent(self):
        return np.arange(self.parent().n_points())


class TSNEEmbedding(Embedding):

    class TSNEWorker(Dataset.Worker):

        def __init__(self, parent, **parameters):
            super().__init__()
            self.parent = parent
            self.parameters = parameters

        def work(self):
            # Hide hidden features
            X_use, X_hidden = self.parent.data(split_hidden=True)

            # t-SNE embedding
            tsne = TSNE(**self.parameters)
            Y_use = tsne.fit_transform(X_use)

            # Restore original hidden features
            Y = np.concatenate((Y_use, X_hidden), axis=1)

            self.ready.emit(Y)

    def __init__(self, parent, name=None, hidden=None, **parameters):
        if 'n_components' not in parameters:
            parameters['n_components'] = 2

        super().__init__(parent, parameters['n_components'], name=name, hidden=hidden)

        worker = TSNEEmbedding.TSNEWorker(parent, **parameters)

        self.spawn_thread(worker, self.set_data, waitfor=(parent,))


class PCAEmbedding(Embedding):

    def __init__(self, parent, name=None, hidden=None, **parameters):
        if 'n_components' not in parameters:
            parameters['n_components'] = 2

        super().__init__(parent, parameters['n_components'], name=name, hidden=hidden)

        worker = PCAEmbedding.PCAWorker(parent, **parameters)

        self.spawn_thread(worker, self.set_data, waitfor=(parent,))

    class PCAWorker(Dataset.Worker):

        def __init__(self, parent, **parameters):
            super().__init__()
            self.parent = parent
            self.parameters = parameters

        def work(self):
            # Hide hidden features
            X_use, X_hidden = self.parent.data(split_hidden=True)

            # PCA embedding
            pca = PCA(**self.parameters)
            Y_use = pca.fit_transform(X_use)

            # Restore original hidden features
            Y = np.concatenate((Y_use, X_hidden), axis=1)

            self.ready.emit(Y)


class MDSEmbedding(Embedding):

    class MDSWorker(Dataset.Worker):

        def __init__(self, parent, **parameters):
            super().__init__()
            self.parent = parent
            self.parameters = parameters

        def work(self):
            # Hide hidden features
            X_use, X_hidden = self.parent.data(split_hidden=True)

            # MDS embedding
            mds = MDS(**self.parameters)
            Y_use = mds.fit_transform(X_use)

            # Restore original hidden features
            Y = np.concatenate((Y_use, X_hidden), axis=1)

            self.ready.emit(Y)

    def __init__(self, parent, name=None, hidden=None, **parameters):
        if 'n_components' not in parameters:
            parameters['n_components'] = 2

        super().__init__(parent, parameters['n_components'], name=name, hidden=hidden)

        worker = MDSEmbedding.MDSWorker(parent, **parameters)

        self.spawn_thread(worker, self.set_data, waitfor=(parent,))


class LAMPEmbedding(Embedding):

    class LAMPWorker(Dataset.Worker):

        def __init__(self, parent, representatives_nd, representatives_2d, **parameters):
            super().__init__()
            self.parent = parent
            self.representatives_nd = representatives_nd
            self.representatives_2d = representatives_2d
            self.parameters = parameters

        def work(self):
            # Hide hidden features

            X_s, _ = self.representatives_nd.data(split_hidden=True)
            Y_s, _ = self.representatives_2d.data(split_hidden=True)
            X_use, X_hidden = self.parent.data(split_hidden=True)

            N = X_use.shape[0]
            n = Y_s.shape[1]

            Y = np.zeros((N, n))

            for i in np.arange(N):
                x = X_use[i, :]

                alphas = ((X_s - x)**-2).sum(axis=1)

                x_tilde = alphas.dot(X_s) / alphas.sum()
                y_tilde = alphas.dot(Y_s) / alphas.sum()

                alphas_T = alphas[:, np.newaxis]
                A = alphas_T**0.5 * (X_s - x_tilde)
                B = alphas_T**0.5 * (Y_s - y_tilde)

                U, _, V = np.linalg.svd(A.T.dot(B), full_matrices=False)

                Y[i, :] = (x - x_tilde).dot(U.dot(V)) + y_tilde

            # Restore original hidden features
            Y = np.concatenate((Y, X_hidden), axis=1)

            self.ready.emit(Y)

    def __init__(self, parent, representatives_2d, name=None, hidden=None, **parameters):
        if name is None:
            name = 'E({}, {})'.format(parent.name(), representatives_2d.name())
        if 'n_components' not in parameters:
            parameters['n_components'] = 2

        super().__init__(parent, parameters['n_components'], name=name, hidden=hidden)

        representatives_nd = RootSelection(representatives_2d)

        worker = LAMPEmbedding.LAMPWorker(parent, representatives_nd, representatives_2d, **parameters)
        self.spawn_thread(worker, self.set_data, waitfor=(parent, representatives_2d, representatives_nd))