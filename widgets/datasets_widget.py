from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import time
from model import *
from widgets import Slider, FileDropWidget
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
        self.keep_representatives_checkbox = QCheckBox('Keep representatives')
        parameters_layout.addWidget(self.keep_representatives_checkbox)
        reproject_button = QPushButton('Reproject')
        reproject_button.clicked.connect(self.reproject_current_view)
        parameters_layout.addWidget(reproject_button)
        for _, slider in self.sliders.items():
            parameters_layout.addLayout(slider)

        self.zo_continuity_checkbox = QCheckBox('Zoom-out continuity')
        self.zo_continuity_checkbox.setChecked(True)
        parameters_layout.addWidget(self.zo_continuity_checkbox)

        parameters_group.setLayout(parameters_layout)

        dataset_drop_box = FileDropWidget('Dataset input', 'Drop to load as dataset', self.read_dataset_from)
        labels_drop_box = FileDropWidget('Labels input', 'Drop to load as labels', self.read_labels_from)
        drops_hbox = QHBoxLayout()
        drops_hbox.addWidget(dataset_drop_box)
        drops_hbox.addWidget(labels_drop_box)

        self.vbox_main = QVBoxLayout()
        self.vbox_main.addWidget(controls_group)
        self.vbox_main.addWidget(parameters_group)
        self.vbox_main.addLayout(drops_hbox)
        self.setLayout(self.vbox_main)

    def nd_zoom_in(self, mouse_pos, verbose=1):
        zoom_fraction = self.get('zoom_fraction')
        repr_fraction = self.get('repr_fraction')
        N_max = self.get('N_max')
        
        t_0 = time.time()

        # Get the fraction of closest points to the mouse.
        the_union = self.dataset_view_renderer.current_union()
        query_2d = knn_selection(the_union, n_samples=round(the_union.n_points() * zoom_fraction), pos=mouse_pos)

        # Use a fraction of the closest points as new representatives
        n_repr = round(repr_fraction * query_2d.n_points() / zoom_fraction)
        representatives_2d = query_2d.random_sampling(n_repr)
        representatives_nd = root_selection(representatives_2d)

        # Fetch new points from the big dataset
        n_samples = N_max - query_2d.n_points()
        query_nd = root_selection(query_2d)
        knn_nd = Dataset.root.knn_pointset(query_dataset=query_nd, n_samples=n_samples, remove_query_points=True)

        # Make sure all points from the query are also in the result. (Continuity)
        closest_nonrepresentatives_2d = query_2d - representatives_2d
        closest_nonrepresentatives_nd = root_selection(closest_nonrepresentatives_2d)
        nonrepresentatives_nd = knn_nd + closest_nonrepresentatives_nd

        # Make LAMP embedding of the fetched points, and the non-representative points from the previous frame,
        # using the picked representatives as fixed representatives.
        new_neighbours_2d = lamp_projection(nonrepresentatives_nd, representatives_nd, representatives_2d)
        
        if verbose:
            print('nD zoom-in took {:.2f} seconds.\n'.format(time.time() - t_0))        

        # Schedule that the projection is shown.
        self.dataset_view_renderer.interpolate_to_dataset(new_neighbours_2d, representatives_2d)

    def nd_zoom_out(self):
        N_max = self.get('N_max')
        k = round(self.get('zoom_fraction')**(-1) * N_max)
        n_repr = round(N_max * self.get('repr_fraction'))

        query_2d = self.dataset_view_renderer.current_union()
        query_nd = root_selection(query_2d)

        # knn_zo_nd = knn_fetching_zo_1(query_nd, 1000, N_max)
        # knn_zo_nd = knn_fetching_zo_2_1(query_nd, N_max=N_max, zoom_factor=1.2)
        knn_zo_nd = knn_fetching_zo_2_2(query_nd, N_max=N_max, zoom_factor=1.2)
        # knn_zo_nd = knn_fetching_zo_4(query_nd, N_max, M=1000)
        
        if self.zo_continuity_checkbox.isChecked():
            representatives_2d = query_2d.random_sampling(n_repr)
            representatives_nd = root_selection(representatives_2d)
        else:
            representatives_nd = knn_zo_nd.random_sampling(n_repr)
            representatives_2d = mds_projection(representatives_nd)
        
        nonrepresentatives_nd = knn_zo_nd - representatives_nd
        nonrepresentatives_2d = lamp_projection(nonrepresentatives_nd, representatives_nd, representatives_2d)

        self.dataset_view_renderer.interpolate_to_dataset(nonrepresentatives_2d, representatives_2d)

    def reproject_current_view(self):
        current_view = self.dataset_view_renderer.current_view()

        if self.keep_representatives_checkbox.isChecked():
            # Get the representative and regular datasets from the current view
            old_representatives_2d = current_view.new_representative()
            old_nonrepresentatives_2d = current_view.new_regular()
            
            # Get the nd data corresponding to the 2D points
            representatives_nd = root_selection(old_representatives_2d)
            nonrepresentatives_nd = root_selection(old_nonrepresentatives_2d)
        else:
            N_max = self.get('N_max')
            n_repr = round(N_max * self.get('repr_fraction'))

            the_union = self.dataset_view_renderer.current_union()
            the_union_nd = root_selection(the_union)

            representatives_nd = the_union_nd.random_sampling(n_repr)
            nonrepresentatives_nd = the_union_nd - representatives_nd

        # Reproject the nd datapoints
        representatives_2d = mds_projection(representatives_nd)
        nonrepresentatives_2d = lamp_projection(nonrepresentatives_nd, representatives_nd, representatives_2d)

        self.dataset_view_renderer.interpolate_to_dataset(nonrepresentatives_2d, representatives_2d)

    @pyqtSlot(object, object)
    def handle_reader_results(self, X, labels):
        # Make the input dataset object
        dataset = Dataset(X, np.arange(X.shape[0]), name='input', is_root=True)
        
        n_points = self.get('N_max')
        repr_fraction = self.get('repr_fraction')

        # Subsample the dataset, if necessary.
        if dataset.n_points() > n_points:
            dataset = dataset.random_sampling(n_points)

        n_repr = round(repr_fraction * dataset.n_points())

        # Subsample the dataset for representatives.
        representatives_nd = dataset.random_sampling(n_repr)
        nonrepresentatives_nd = dataset - representatives_nd

        if dataset.n_dimensions() > 2:
            # Project with MDS+LAMP
            representatives_2d = mds_projection(representatives_nd)
            nonrepresentatives_2d = lamp_projection(nonrepresentatives_nd, representatives_nd, representatives_2d)
        elif dataset.n_dimensions() == 2:
            # Don't project, since dimensionality is 2.
            representatives_2d = representatives_nd
            nonrepresentatives_2d = nonrepresentatives_nd

        self.dataset_view_renderer.show_dataset(nonrepresentatives_2d, representatives_2d, fit_to_view=True)

        # Set labels, if they were given in the same file.
        if labels is not None:
            success = Dataset.set_root_labels(labels)
            if success:
                self.imp_window.visuals_widget.add_colour_option('Label', Dataset.labels)

    def handle_label_reader_results(self, labels, notgiven):
        if notgiven is not None:
            labels = np.column_stack([labels, notgiven])
            print('Warning, added hidden dimensions to labels.')
        
        success = Dataset.set_root_labels(labels)
        if success:
            self.imp_window.visuals_widget.add_colour_option('Label', Dataset.labels)

    def read_dataset_from(self, path):
        self.reader = Reader(path)
        self.reader.start(ready=self.handle_reader_results)

    def read_labels_from(self, path):
        self.label_reader = Reader(path)
        self.label_reader.start(ready=self.handle_label_reader_results)

    def get(self, name):
        return self.sliders[name].value()
