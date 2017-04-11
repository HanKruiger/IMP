from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class FileDropWidget(QGroupBox):

    def __init__(self, name, hint, callback):
        super().__init__(name)

        self.hint_label = QLabel(hint)
        self.hint_label.setVisible(False)
        hbox = QHBoxLayout()
        hbox.addWidget(self.hint_label)
        self.setLayout(hbox)

        self.callback = callback

        self.setAcceptDrops(True)

    def dragEnterEvent(self, drag_enter_event):
        if drag_enter_event.mimeData().hasUrls():
            urls = drag_enter_event.mimeData().urls()
            if not all([url.isValid() for url in urls]):
                print('Invalid URL(s): {0}'.format([url.toString() for url in urls if not url.isValid()]))
            elif not all([url.isLocalFile() for url in urls]):
                print('Non-local URL(s): {0}'.format([url.toString() for url in urls if not url.isLocalFile()]))
            elif len(urls) > 1:
                print('Dropping more than one file is not supported anymore.')
            else:
                self.hint_label.setVisible(True)
                drag_enter_event.acceptProposedAction()

    def dragLeaveEvent(self, drag_leave_event):
        self.hint_label.setVisible(False)
        drag_leave_event.accept()

    def dropEvent(self, drop_event):
        self.hint_label.setVisible(False)
        urls = drop_event.mimeData().urls()
        path = urls[0].path()
        self.callback(path)
        drop_event.accept()
