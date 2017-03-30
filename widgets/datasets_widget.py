from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *
from widgets import Slider

import numpy as np


class DatasetsWidget(QWidget):

    def __init__(self, imp_window):
        super().__init__()
        self.imp_window = imp_window
        self.dataset_view_renderer = self.imp_window.gl_widget.dataset_view_renderer

        controls_group = QGroupBox('History')
        row = QHBoxLayout()
        previous_view_button = QPushButton('Back')
        previous_view_button.clicked.connect(self.dataset_view_renderer.previous)
        row.addWidget(previous_view_button)
        next_view_button = QPushButton('Forward')
        next_view_button.clicked.connect(self.dataset_view_renderer.next)
        row.addWidget(next_view_button)
        controls_group.setLayout(row)

        self.sliders = dict()
        self.sliders['N_max'] = Slider('Maximum number of points', 50, 3000, 1000, data_type=int)
        self.sliders['repr_max'] = Slider('Maximum number of representatives', 20, 100, 30, data_type=int)
        self.sliders['zoom_fraction'] = Slider('Zoom fraction (%)', 1, 99, 50, data_type=int)

        parameters_group = QGroupBox('Projection control')
        parameters_layout = QVBoxLayout()
        reproject_button = QPushButton('Reproject')
        reproject_button.clicked.connect(self.reproject_current_view)
        parameters_layout.addWidget(reproject_button)
        for _, slider in self.sliders.items():
            parameters_layout.addLayout(slider)
            
        parameters_group.setLayout(parameters_layout)

        self.vbox_main = QVBoxLayout()
        self.vbox_main.addWidget(controls_group)
        self.vbox_main.addWidget(parameters_group)
        self.setLayout(self.vbox_main)

        self.setAcceptDrops(True)

    def hierarchical_zoom(self, mouse_pos, zoomin=True):
        fraction = self.get('zoom_fraction') / 100
        N_max = self.get('N_max')
        if zoomin:
            # Get the 90% closest points to the mouse.
            union = self.dataset_view_renderer.current_union()
            closest = KNNSampling(union, n_samples=round(union.n_points() * fraction), pos=mouse_pos)
            # visibles, invisibles = self.dataset_view_renderer.filter_unseen_points()

            # if visibles is not None and invisibles is not None:
            n_repr = self.get('repr_max')
            # visible_points = Selection(union, idcs=visibles)
            if closest.n_points() > n_repr:
                representatives_2d = RandomSampling(closest, n_repr)
            else:
                assert(False)
                representatives_2d = closest

            n_fetch = N_max - representatives_2d.n_points()
            knn_fetching = KNNFetching(representatives_2d, n_fetch)

            new_neighbours_2d = LAMPEmbedding(knn_fetching, representatives_2d)
            new_neighbours_2d.ready.connect(
                lambda: self.dataset_view_renderer.interpolate_to_dataset(new_neighbours_2d, representatives_2d)
            )
        else:
            raise NotImplementedError

    def reproject_current_view(self):
        current_view = self.dataset_view_renderer.current_view()
        representatives = current_view.new_representative()
        regulars = current_view.new_regular()
        representatidves_nd = RootSelection(representatives)
        regulars_nd = RootSelection(regulars)
        representatives_reprojected = MDSEmbedding(representatidves_nd)
        regulars_reprojected = LAMPEmbedding(regulars_nd, representatives_reprojected)
        regulars_reprojected.ready.connect(
            lambda: self.dataset_view_renderer.interpolate_to_dataset(regulars_reprojected, representatives_reprojected)
        )

    @pyqtSlot(object)
    def handle_reader_results(self, dataset):
        self.imp_window.statusBar().clearMessage()
        N_max = self.get('N_max')
        n_samples = self.get('repr_max')
        if dataset.n_points() > N_max:
            sampling = RandomSampling(dataset, N_max)
            dataset = sampling

        representatives_2d = None
        if dataset.n_dimensions(count_hidden=False) > 2:
            representatives_nd = RandomSampling(dataset, n_samples)
            representatives_2d = MDSEmbedding(representatives_nd, n_components=2)

            dataset_diff = Difference(dataset, representatives_nd)
            dataset_emb = LAMPEmbedding(dataset_diff, representatives_2d)
            dataset_emb.ready.connect(
                lambda: self.dataset_view_renderer.show_dataset(dataset_emb, representatives_2d, fit_to_view=True)
            )

    def remove_dataset(self, dataset):
        self.dataset_view_renderer.remove_dataset(dataset)

        dataset.destroy()

    def dragEnterEvent(self, drag_enter_event):
        if drag_enter_event.mimeData().hasUrls():
            urls = drag_enter_event.mimeData().urls()
            if not all([url.isValid() for url in urls]):
                qDebug('Invalid URL(s): {0}'.format([url.toString() for url in urls if not url.isValid()]))
            elif not all([url.isLocalFile() for url in urls]):
                qDebug('Non-local URL(s): {0}'.format([url.toString() for url in urls if not url.isLocalFile()]))
            else:
                self.imp_window.statusBar().showMessage('Drop to load {0} as new dataset'.format(', '.join([url.fileName() for url in urls])))
                drag_enter_event.acceptProposedAction()

    def dragLeaveEvent(self, drag_leave_event):
        self.imp_window.statusBar().clearMessage()
        drag_leave_event.accept()

    def dropEvent(self, drop_event):
        urls = drop_event.mimeData().urls()
        paths = [url.path() for url in urls]

        dataset = InputDataset(paths)
        # self.add_dataset(dataset)
        dataset.data_ready.connect(self.handle_reader_results)

        drop_event.accept()

        # Set focus to our window after the drop event.
        self.activateWindow()

        self.imp_window.statusBar().showMessage('Loading {0}...'.format([url.fileName() for url in urls]))

    def get(self, name):
        return self.sliders[name].value()
