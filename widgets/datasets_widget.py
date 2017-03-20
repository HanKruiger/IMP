from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model import *

import numpy as np

class DatasetsWidget(QGroupBox):

    def __init__(self, imp_window):
        super().__init__('Datasets')
        self.imp_window = imp_window
        self.dataset_view_renderer = self.imp_window.gl_widget.dataset_view_renderer

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Name', 'N', 'm'])
        self.model.dataChanged.connect(self.data_changed)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_menu)
        self.tree_view.doubleClicked.connect(self.show_dataset_item)

        # Resize column widths
        for i in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(i)
        
        params_vbox = QVBoxLayout()
        self.N_max_textbox = QLineEdit()
        self.N_max_textbox.setValidator(QIntValidator())
        self.N_max_textbox.setText(str(500))
        params_vbox.addWidget(QLabel('Maximum number of points'))
        params_vbox.addWidget(self.N_max_textbox)

        self.repr_max_textbox = QLineEdit()
        self.repr_max_textbox.setValidator(QIntValidator())
        self.repr_max_textbox.setText(str(50))
        params_vbox.addWidget(QLabel('Maximum number of representatives'))
        params_vbox.addWidget(self.repr_max_textbox)

        self.vbox_main = QVBoxLayout()
        self.vbox_main.addWidget(self.tree_view)
        self.vbox_main.addLayout(params_vbox)
        self.setLayout(self.vbox_main)

        self.setAcceptDrops(True)

    def datasets(self):
        datasets = set()

        def add_to_set(dataset_item, datasets):
            if not isinstance(dataset_item, DatasetItem):
                return
            datasets.add(dataset_item.data(role=Qt.UserRole))
            if dataset_item.hasChildren():
                for row in range(dataset_item.rowCount()):
                    for col in range(dataset_item.columnCount()):
                        add_to_set(dataset_item.child(row, col), datasets)

        for idx in range(self.model.rowCount()):
            dataset = self.model.data(self.model.index(idx, 0), role=Qt.UserRole)
            dataset_item = dataset.q_item()
            add_to_set(dataset_item, datasets)

        return datasets

    def data_changed(self, topleft, bottomright, roles):
        dataset = self.model.data(topleft, role=Qt.UserRole)

        # If there's no user role, it's not the right column.
        if dataset is None:
            return

        # Read potentially new name from display role
        new_name = self.model.data(topleft, role=Qt.DisplayRole)
        dataset.set_name(new_name)
        
        # Make new entries reflect new data dimensions/size
        self.model.setData(
            self.model.index(topleft.row(), 1, self.model.parent(topleft)), dataset.n_points()
        )
        self.model.setData(
            self.model.index(topleft.row(), 2, self.model.parent(topleft)), dataset.n_dimensions()
        )

    @pyqtSlot(QModelIndex)
    def show_dataset_item(self, item):
        dataset = self.model.data(item, role=Qt.UserRole)
        if dataset is None:
            return
        self.dataset_view_renderer.show_dataset(dataset, fit_to_view=False)
        

    def hierarchical_zoom(self, zoomin=True):
        if zoomin:
            visibles, invisibles, union = self.dataset_view_renderer.filter_unseen_points()

            if visibles is not None and invisibles is not None:
                n_samples = int(self.repr_max_textbox.text())
                N_max = int(self.N_max_textbox.text())
                selection = Selection(union, idcs=visibles)
                if n_samples < selection.n_points():
                    representatives_2d = RandomSampling(selection, n_samples)
                else:
                    representatives_2d = selection

                knn_fetching = KNNFetching(selection, N_max - n_samples)

                # The KNN fetch MINUS the representatives.
                new_neighbours_nd = Difference(knn_fetching, representatives_2d)

                new_neighbours_2d = LAMPEmbedding(new_neighbours_nd, representatives_2d)
                new_neighbours_2d.ready.connect(
                    lambda: self.dataset_view_renderer.show_dataset(new_neighbours_2d, representatives_2d)
                )
        else:
            raise NotImplementedError

    @pyqtSlot(object)
    def handle_reader_results(self, dataset):
        self.imp_window.statusBar().clearMessage()
        N_max = int(self.N_max_textbox.text())
        n_samples = int(self.repr_max_textbox.text())
        if dataset.n_points() > N_max:
            sampling = RandomSampling(dataset, N_max)
            dataset = sampling

        representatives_2d = None
        if dataset.n_dimensions(count_hidden=False) > 2:
            representatives_nd = RandomSampling(dataset, n_samples)
            representatives_2d = MDSEmbedding(representatives_nd, n_components=2)
            
            dataset_diff = Difference(dataset, representatives_nd)
            dataset_emb = LAMPEmbedding(dataset, representatives_2d)
            dataset_emb.ready.connect(
                lambda: self.dataset_view_renderer.show_dataset(dataset_emb, representatives_2d, fit_to_view=True)
            )
        

    @pyqtSlot(object)
    def add_dataset(self, dataset):
        dataset.has_new_child.connect(self.add_dataset)

        dataset_item = DatasetItem(dataset.name())
        dataset_item.setData(dataset, role=Qt.UserRole)

        N_item = QStandardItem(str(dataset.n_points()))
        N_item.setEditable(False)
        m_item = QStandardItem(str(dataset.n_dimensions()))
        m_item.setEditable(False)

        if dataset.parent() is None:
            # Add it to the model's root node.
            self.model.appendRow([dataset_item, N_item, m_item])
        else:
            # Fetch the parent DatasetItem
            parent_item = dataset.parent().q_item()

            # Append the new dataset's row to the parent.
            parent_item.appendRow([dataset_item, N_item, m_item])

            # Expand the parent. (This does not happen automatically)
            self.tree_view.expand(parent_item.index())

        # Resize column widths based on new contents
        for i in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(i)

        self.imp_window.statusBar().showMessage('Added dataset.', msecs=2000)

    def remove_dataset(self, dataset):
        self.dataset_view_renderer.remove_dataset(dataset)

        dataset.destroy()
        if dataset.parent() is not None:
            self.model.removeRows(dataset.q_item().row(), 1, dataset.parent().q_item().index())
        else:
            self.model.removeRows(dataset.q_item().row(), 1)

    def open_menu(self, position):
        indexes = self.tree_view.selectedIndexes()
        if len(indexes) != self.model.columnCount():
            return

        # Get the Dataset object from the model
        dataset = self.model.data(indexes[0], role=Qt.UserRole)

        # Build menu, based on characteristics of Dataset object.
        menu = QMenu()

        if dataset.child_count() == 0:
            delete_action = menu.addAction('Delete')
            delete_action.triggered.connect(lambda: self.remove_dataset(dataset))

        # Makes sure the menu pops up where the mouse pointer is.
        menu.exec_(self.tree_view.viewport().mapToGlobal(position))

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

    def dropEvent(self, drop_event):
        urls = drop_event.mimeData().urls()
        paths = [url.path() for url in urls]

        dataset = InputDataset(paths)
        self.add_dataset(dataset)
        dataset.data_ready.connect(self.handle_reader_results)

        self.imp_window.statusBar().showMessage('Loading {0}...'.format([url.fileName() for url in urls]))

