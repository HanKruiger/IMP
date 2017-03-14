from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *
from operators.utils import hide_features
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np


class Embedding(Dataset):

    def __init__(self, parent, name=None, hidden=None):
        if name is None:
            name = 'E({})'.format(parent.name())
        super().__init__(name, parent, data=None, hidden=hidden)

    def root(self):
        return self.parent().root()


class TSNEEmbedding(Embedding):

    class TSNEWorker(Worker):

        def __init__(self, parent, **parameters):
            super().__init__()
            self.parent = parent
            self.parameters = parameters

        def work(self):
            # Hide hidden features
            X_use, X_hidden = hide_features(self.parent.data(), self.parent.hidden_features())

            # t-SNE embedding
            tsne = TSNE(**self.parameters)
            Y_use = tsne.fit_transform(X_use)

            # Restore original hidden features
            Y = np.concatenate((Y_use, X_hidden), axis=1)

            self.ready.emit(Y)

    def __init__(self, parent, name=None, hidden=None, **parameters):
        super().__init__(parent, name=name, hidden=hidden)

        worker = TSNEEmbedding.TSNEWorker(parent, **parameters)

        self.spawn_thread(worker, self.set_data, waitfor=(parent,))


class PCAEmbedding(Embedding):

    class PCAWorker(Worker):

        def __init__(self, parent, **parameters):
            super().__init__()
            self.parent = parent
            self.parameters = parameters

        def work(self):
            # Hide hidden features
            X_use, X_hidden = hide_features(self.parent.data(), self.parent.hidden_features())

            # PCA embedding
            pca = PCA(**self.parameters)
            Y_use = pca.fit_transform(X_use)

            # Restore original hidden features
            Y = np.concatenate((Y_use, X_hidden), axis=1)

            self.ready.emit(Y)

    def __init__(self, parent, name=None, hidden=None, **parameters):
        super().__init__(parent, name=name, hidden=hidden)

        if 'n_components' not in parameters:
            parameters['n_components'] = 2

        worker = PCAEmbedding.PCAWorker(parent, **parameters)

        self.spawn_thread(worker, self.set_data, waitfor=(parent,))


class LAMPEmbedding(Embedding):

    class LAMPWorker(Worker):

        def __init__(self, parent, representatives_nd, representatives_2d, **parameters):
            super().__init__()
            self.parent = parent
            self.representatives_nd = representatives_nd
            self.representatives_2d = representatives_2d
            self.parameters = parameters

        def work(self):
            # Hide hidden features
            X_s, _ = hide_features(self.representatives_nd.data(), self.representatives_nd.hidden_features())
            Y_s, _ = hide_features(self.representatives_2d.data(), self.representatives_2d.hidden_features())
            X_use, X_hidden = hide_features(self.parent.data(), self.parent.hidden_features())

            N = X.shape[0]
            n = Y_s.shape[1]

            Y = np.zeros((N, n))

            for i in np.arange(N):
                x = X[i, :]

                alphas = np.sum((X_s - x)**2, axis=1)

                x_tilde = (alphas * X_s.T).sum(axis=1) / alphas.sum()
                y_tilde = (alphas * Y_s.T).sum(axis=1) / alphas.sum()

                A_T = alphas * (X_s - x_tilde).T
                B = (alphas * (Y_s - y_tilde).T).T

                U, _, V = np.linalg.svd(A_T.dot(B), full_matrices=False)

                Y[i, :] = (x - x_tilde).dot(U.dot(V)) + y_tilde

            # Restore original hidden features
            Y = np.concatenate((Y_use, X_hidden), axis=1)

            self.ready.emit(Y)

    def __init__(self, parent, representatives_2d, name=None, hidden=None, **parameters):
        if name is None:
            name = 'E({}, {})'.format(parent.name(), representatives_2d.name())
        super().__init__(parent, name=name, hidden=hidden)

        representatives_nd = RootSelection(representatives_2d)

        worker = LAMPEmbedding.LAMPWorker(parent, representatives_nd, representatives_2d, parameters)

        self.spawn_thread(worker, self.set_data, waitfor=(parent, representatives_2d, representatives_nd))
