from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

import numpy as np

from model.dataset import Dataset, Merging
from operators.operator import Operator
from operators.embedders import LAMPEmbedder
from operators.samplers import RandomSampler, SVDBasedSampler
from operators.selectors import LenseSelector
from operators.fetchers import HypersphereFetcher

class HierarchicalZoom(QObject):

    when_done = pyqtSignal(object, object)

    def __init__(self, parent, selected, center, radius, x_dim, y_dim, N_max):
        super().__init__()
        self.N_max = N_max
        self.prepare_selection(parent, selected, center, radius, x_dim, y_dim)

    def prepare_selection(self, parent, selected, center, radius, x_dim, y_dim):
        self.selector = LenseSelector()
        self.selector.set_input({
            'embedding': selected,
            'parent': parent
        })
        self.selector.set_parameters({
            'center': center,
            'radius': radius,
            'x_dim': x_dim,
            'y_dim': y_dim
        })
        self.parent = parent

    def run(self):
        self.select()

    def select(self):
        self.parent.operation_finished.connect(self.set_selection_results)
        self.parent.perform_operation(self.selector)

    @pyqtSlot(object, object)
    def set_selection_results(self, result, operator):
        if operator != self.selector:
            return # This was not the operator I was waiting for.
        
        # Unsubscribe from further events on dataset
        result[0].parent().operation_finished.disconnect(self.set_selection_results)
        
        self.query_nd = result[0]
        self.query_2d = result[1]
        
        self.fetch()

    def fetch(self):
        # Find the root dataset for this dataset
        self.root_dataset = self.parent
        while self.root_dataset.parent() is not None and not isinstance(self.root_dataset, Merging):
            self.root_dataset = self.root_dataset.parent()

        self.fetcher = HypersphereFetcher()
        self.fetcher.set_input({
            'nd_dataset': self.root_dataset,
            'query_nd': self.query_nd,
            'query_2d': self.query_2d
        })
        self.fetcher.set_parameters({
            'center': self.selector.parameters()['center'],
            'radius': self.selector.parameters()['radius']    
        })

        self.root_dataset.operation_finished.connect(self.set_fetch_results)
        self.root_dataset.perform_operation(self.fetcher)
        
    # Subsample operation, performed after the fetching, if needed.
    @pyqtSlot(object, object)
    def set_fetch_results(self, result, operator):
        if operator != self.fetcher:
            return # This was not the operator I was waiting for.
        
        # Unsubscribe from further events on dataset
        self.root_dataset.operation_finished.disconnect(self.set_fetch_results)

        self.fetch_results = result[0]
        
        self.subsample_fetch_results()

    def subsample_fetch_results(self):
        if self.fetch_results.N > self.N_max:
            self.subsampler = RandomSampler()
            self.subsampler.set_input({
                'parent': self.fetch_results
            })
            self.subsampler.set_parameters({
                'n_samples': self.N_max
            })
            self.fetch_results.operation_finished.connect(self.set_subsampled_fetch_results)
            self.fetch_results.perform_operation(self.subsampler)
        else:
            print('No need to subsample fetched results. Hit lowest-level data. ({} points)'.format(self.fetch_results.N))
            self.subsampler = None
            self.set_subsampled_fetch_results((self.fetch_results,), None)

    @pyqtSlot(object, object)
    def set_subsampled_fetch_results(self, result, operator):
        if operator != self.subsampler:
            return # This was not the operator I was waiting for.
        if operator is not None:
            # Unsubscribe from further events on dataset
            result[0].parent().operation_finished.disconnect(self.set_subsampled_fetch_results)

        self.subsampled_fetch_results = result[0]

        self.subsample_representatives()

    def subsample_representatives(self):
        if self.parent.N == self.query_nd.N:
            # It is a zoom OUT operation. Keep fraction of control points.
            n_keep = int(self.query_nd.N * (self.query_nd.N / self.fetch_results.N))
            print('Zoom OUT: Keeping {} of the {} selected 2D points as control points.'.format(n_keep, self.query_nd.N))

            self.repr_subsampler = RandomSampler()
            self.repr_subsampler.set_input({
                'parent': self.query_nd,
                'sibling': self.query_2d
            })
            self.repr_subsampler.set_parameters({
                'n_samples': n_keep
            })
            
            self.query_nd.operation_finished.connect(self.set_representatives)
            self.query_nd.perform_operation(self.repr_subsampler)
        elif self.parent.N > self.query_nd.N:
            print('Zoom IN: Keeping selected points as control points.')
            # It is a zoom IN operation. Use points in the selection as control points.
            self.repr_subsampler = None
            self.set_representatives((self.query_nd,self.query_2d), None)

    @pyqtSlot(object, object)
    def set_representatives(self, result, operator):
        if operator != self.repr_subsampler:
            return # This was not the operator I was waiting for.
        if operator is not None:
            # Unsubscribe from further events on dataset
            result[0].parent().operation_finished.disconnect(self.set_representatives)

        self.representatives_nd = result[0]
        self.representatives_2d = result[1]

        self.embed()

    # Embed operation
    def embed(self):
        self.embedder = LAMPEmbedder()
        self.embedder.set_input({
            'parent': self.subsampled_fetch_results,
            'representatives_nd': self.representatives_nd,
            'representatives_2d': self.representatives_2d
        })
        self.embedder.set_parameters({
            'n_hidden_features': self.subsampled_fetch_results.hidden_features()
        })

        self.subsampled_fetch_results.operation_finished.connect(self.set_embed_results)
        self.subsampled_fetch_results.perform_operation(self.embedder)

    # Embed operation
    @pyqtSlot(object, object)
    def set_embed_results(self, result, operator):
        if operator != self.embedder:
            return # This was not the operator I was waiting for.
        # Unsubscribe from further events on dataset
        result[0].parent().operation_finished.disconnect(self.set_embed_results)
        
        self.when_done.emit(result[0], self)
