from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *
from widgets import Slider
from operators import *

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
        self.sliders['repr_fraction'] = Slider('Representatives fraction', 0.01, 0.99, 0.05, data_type=float)
        self.sliders['zoom_fraction'] = Slider('Zoom fraction', 0.01, 0.99, 0.50, data_type=float)

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
        zoom_fraction = self.get('zoom_fraction')
        repr_fraction = self.get('repr_fraction')
        N_max = self.get('N_max')
        
        if zoomin:
            # Get the fraction of closest points to the mouse.
            the_union = self.dataset_view_renderer.current_union()
            closest = knn_selection(the_union, n_samples=round(the_union.n_points() * zoom_fraction), pos=mouse_pos)

            # Use a fraction of the closest points as new representatives
            n_repr = round(repr_fraction * closest.n_points() / zoom_fraction)
            representatives_2d = random_sampling(closest, n_repr)
            representatives_nd = root_selection(representatives_2d)

            # Fetch new points from the big dataset
            n_fetch = N_max - closest.n_points()
            nd_closest = knn_fetching(Dataset.root, closest, n_fetch)

            # Remove the representatives from the closest points, and get the nd data of those
            closest_diff = difference(closest, representatives_2d)
            closest_diff_root = root_selection(closest_diff)

            # Make LAMP embedding of the fetched points, and the non-representative points from the previous frame,
            # using the picked representatives as fixed representatives.
            new_neighbours_2d = lamp_projection(union(nd_closest, closest_diff_root), representatives_nd, representatives_2d)
            
            # Schedule that the projection is shown.
            self.dataset_view_renderer.interpolate_to_dataset(new_neighbours_2d, representatives_2d)
        else:
            raise NotImplementedError

    def reproject_current_view(self):
        current_view = self.dataset_view_renderer.current_view()

        # Get the representative and regular datasets from the current view
        representative = current_view.new_representative()
        regular = current_view.new_regular()
        
        # Get the nd data corresponding to the 2D points
        representative_nd = root_selection(representative)
        regular_nd = root_selection(regular)

        # Reproject the nd datapoints
        representative_reprojected = mds_projection(representative_nd)
        regular_reprojected = lamp_projection(regular_nd, representative_nd, representative_reprojected)

        self.dataset_view_renderer.interpolate_to_dataset(regular_reprojected, representative_reprojected)

    @pyqtSlot(object)
    def handle_reader_results(self, dataset):
        self.imp_window.statusBar().showMessage('Loading finished. Projecting {}...'.format(dataset.name()))
        n_points = self.get('N_max')
        repr_fraction = self.get('repr_fraction')

        # Subsample the dataset, if necessary.
        if dataset.n_points() > n_points:
            sampling = random_sampling(dataset, n_points)
            dataset = sampling

        n_repr = round(repr_fraction * dataset.n_points())

        # Subsample the subsampled dataset, always.
        representatives_nd = random_sampling(dataset, n_repr)
        dataset_diff_nd = difference(dataset, representatives_nd)

        if dataset.n_dimensions() > 2:
            # Project with MDS+LAMP
            representatives_2d = mds_projection(representatives_nd, n_components=2)
            dataset_diff_2d = lamp_projection(dataset_diff_nd, representatives_nd, representatives_2d)
        elif dataset.n_dimensions() == 2:
            # Don't project, since dimensionality is 2.
            representatives_2d = representatives_nd
            dataset_diff_2d = dataset_diff_nd

        self.dataset_view_renderer.show_dataset(dataset_diff_2d, representatives_2d, fit_to_view=True)

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
        if len(paths) > 1:
            print('Not supported anymore. Drop labels in other bin manually.')
            return
        path = paths[0]

        self.reader = Reader(path)
        self.reader.start(ready=self.handle_reader_results)

        drop_event.accept()
        
        # Set focus to our window after the drop event. (Otherwise focus stays at the source window.)
        self.activateWindow()

        self.imp_window.statusBar().showMessage('Loading {0}...'.format([url.fileName() for url in urls]))

    def get(self, name):
        return self.sliders[name].value()
