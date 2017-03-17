#!/usr/bin/env python3

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys

from widgets import *

if __name__ == '__main__':
    qsf = QSurfaceFormat()
    qsf.setRenderableType(QSurfaceFormat.OpenGL)
    qsf.setProfile(QSurfaceFormat.CoreProfile)
    qsf.setVersion(4, 1)
    qsf.setSamples(4)
    QSurfaceFormat.setDefaultFormat(qsf)

    app = QApplication(sys.argv)

    imp = IMPWindow()

    sys.exit(app.exec_())
