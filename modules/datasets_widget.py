from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from modules.dataset import DatasetItem
from modules.dataset import InputDataset
from modules.clusterers import ClusterReplicator
from modules.projection_dialog import ProjectionDialog
from modules.clustering_dialog import ClusteringDialog


class DatasetsWidget(QGroupBox):

    def __init__(self, imp_app):
        super().__init__('Datasets')
        self.imp_app = imp_app

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Name', 'N', 'm', 'rel'])

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_menu)
        # Resize column widths
        for i in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(i)

        self.vbox_main = QVBoxLayout()
        self.vbox_main.addWidget(self.tree_view)
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
            print(dataset_item)
            add_to_set(dataset_item, datasets)

        return datasets

    @pyqtSlot(object)
    def add_dataset(self, dataset):
        # Connect the dataset's s.t. when it has an embedding, that it adds it here.
        dataset.embedding_finished.connect(self.add_dataset)

        dataset_item = DatasetItem(dataset.name)
        dataset_item.setData(dataset, role=Qt.UserRole)

        N_item = QStandardItem(str(dataset.N))
        N_item.setEditable(False)
        m_item = QStandardItem(str(dataset.m))
        m_item.setEditable(False)
        rel_item = QStandardItem(str(dataset.relation))
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

        if dataset.m > 2:
            projection_dialog = ProjectionDialog(self, dataset, self.imp_app)
            project_action = menu.addAction('Project')
            project_action.triggered.connect(projection_dialog.show)

        clustering_dialog = ClusteringDialog(self, dataset, self.imp_app)
        cluster_action = menu.addAction('Cluster')
        cluster_action.triggered.connect(clustering_dialog.show)

        clustered_datasets = [dataset for dataset in self.datasets() if dataset.is_clustering()]
        matching_clusterings = [clustering for clustering in clustered_datasets if clustering.parent().N == dataset.N]
        if matching_clusterings:
            same_clustering_menu = menu.addMenu('Do same clustering as')
            for clustering in matching_clusterings:
                same_clustering_menu.addAction(clustering.name)

                @pyqtSlot()
                def replicate_clustering():
                    dataset.make_embedding(ClusterReplicator(clustering))

                same_clustering_menu.triggered.connect(replicate_clustering)


        if dataset.m <= 3:
            if not self.imp_app.visuals_widget.is_in_attributes(dataset):
                add_visual_attribute_action = menu.addAction('Add to visual attributes')

                @pyqtSlot()
                def add_to_visual_attributes():
                    self.imp_app.visuals_widget.use_as_attribute(dataset)
                add_visual_attribute_action.triggered.connect(add_to_visual_attributes)
            else:
                remove_visual_attribute_action = menu.addAction('Remove from visual attributes')

                @pyqtSlot()
                def remove_from_visual_attributes():
                    self.imp_app.visuals_widget.remove_as_attribute(dataset)
                remove_visual_attribute_action.triggered.connect(remove_from_visual_attributes)

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
        for url in urls:
            input_dataset = InputDataset(url.path())

            @pyqtSlot()
            def callback():
                self.imp_app.statusBar().clearMessage()
                self.add_dataset(input_dataset)

            input_dataset.data_ready.connect(callback)
            self.imp_app.statusBar().showMessage('Loading {0}...'.format(url.fileName()))
            input_dataset.load_data()
