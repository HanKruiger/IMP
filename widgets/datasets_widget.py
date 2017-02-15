from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from model.dataset import DatasetItem, InputDataset, Selection, Embedding, Dataset
from operators.selectors import LenseSelector
from operators.readers import Reader
from operators.operator_chains import HierarchicalZoom
from widgets.operator_dialog import OperatorDialog


class DatasetsWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Datasets')
        self.imp_app = imp_app

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Name', 'N', 'm', 'rel'])
        self.model.dataChanged.connect(self.data_changed)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_menu)
        self.tree_view.doubleClicked.connect(self.show_dataset)

        self._workers = set()

        # Resize column widths
        for i in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(i)

        self.vbox_main = QVBoxLayout()
        self.vbox_main.addWidget(self.tree_view)
        self.setLayout(self.vbox_main)

        self.setAcceptDrops(True)

    def hierarchical_zoom(self, dataset, x_dim, y_dim, center, radius):
        parent = dataset
        while type(parent) == Embedding:
            parent = parent.parent()


        hierchical_zoom = HierarchicalZoom()
        hierchical_zoom.set_input({
            'parent': parent,
            'selected': dataset
        })
        hierchical_zoom.set_parameters({
            'center': center,
            'radius': radius,
            'x_dim': x_dim,
            'y_dim': y_dim
        })
        self.rememberme = hierchical_zoom
        hierchical_zoom.initialize()
        hierchical_zoom.run()

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
        new_name = self.model.data(topleft, role=Qt.DisplayRole)

        dataset.set_name(new_name)

    # @pyqtSlot(int) Somehow I cannot decorate this!
    def show_dataset(self, item):
        dataset = self.model.data(item, role=Qt.UserRole)
        if dataset is None:
            return

        if not self.imp_app.visuals_widget.current_dataset() == dataset:
            self.imp_app.visuals_widget.update_attribute_list(dataset)
        else:
            self.imp_app.visuals_widget.clear_attributes()

    @pyqtSlot(object)
    def handle_reader_results(self, reader):
        self.imp_app.statusBar().clearMessage()
        for dataset in reader.output():
            self.add_dataset(dataset)
        self._workers.remove(reader)

    @pyqtSlot(object)
    def add_dataset(self, dataset, operator=None):
        if not isinstance(dataset, Dataset):
            for d in dataset:
                self.add_dataset(d, operator)
            return
        # Connect the dataset's s.t. when it has an embedding, that it adds it here.
        dataset.operation_finished.connect(self.add_dataset)

        dataset_item = DatasetItem(dataset.name())
        dataset_item.setData(dataset, role=Qt.UserRole)

        N_item = QStandardItem(str(dataset.N))
        N_item.setEditable(False)
        m_item = QStandardItem(str(dataset.m))
        m_item.setEditable(False)
        rel_item = QStandardItem(type(dataset).__name__)
        rel_item.setEditable(False)

        if dataset.parent() is None:
            # Add it to the model's root node.
            self.model.appendRow([dataset_item, N_item, m_item, rel_item])
        else:
            # Fetch the parent DatasetItem
            parent_item = dataset.parent().q_item()

            # Append the new dataset's row to the parent.
            parent_item.appendRow([dataset_item, N_item, m_item, rel_item])

            # Expand the parent. (This does not happen automatically)
            self.tree_view.expand(parent_item.index())

        # Resize column widths based on new contents
        for i in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(i)

        self.imp_app.statusBar().showMessage('Added dataset.', msecs=2000)

    def remove_dataset(self, dataset):
        if dataset == self.imp_app.visuals_widget.current_dataset():
            self.imp_app.visuals_widget.clear_attributes()

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

        operator_dialog = OperatorDialog(self, dataset, self.imp_app)
        operator_action = menu.addAction('Perform operation')
        operator_action.triggered.connect(operator_dialog.show)

        if dataset.child_count() == 0:
            delete_action = menu.addAction('Delete')

            @pyqtSlot()
            def delete_dataset():
                self.remove_dataset(dataset)

            delete_action.triggered.connect(delete_dataset)

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
                self.imp_app.statusBar().showMessage('Drop to load {0} as new dataset'.format(', '.join([url.fileName() for url in urls])))
                drag_enter_event.acceptProposedAction()

    def dragLeaveEvent(self, drag_leave_event):
        self.imp_app.statusBar().clearMessage()

    def dropEvent(self, drop_event):
        urls = drop_event.mimeData().urls()
        paths = [url.path() for url in urls]

        reader = Reader()
        reader.set_parameters({'paths': paths})
        reader.has_results.connect(self.handle_reader_results)
        reader.start()

        self.imp_app.statusBar().showMessage('Loading {0}...'.format([url.fileName() for url in urls]))
        self._workers.add(reader)

