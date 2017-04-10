from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import abc

class Operator(QObject):

    def __init__(self):
        super().__init__()

    def start(self, ready):
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self.work)

        self.ready.connect(ready)
        self.ready.connect(self.thread.quit)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @abc.abstractmethod
    def work(self):
        """Method that does the operation. (Will be executed in a seperate thread.)"""