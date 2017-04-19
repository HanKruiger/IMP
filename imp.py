#!/usr/bin/env python3

import sys
# Check if we're in a virtual environment.
if sys.prefix == sys.base_prefix:
    from warnings import warn
    warn('Not using a virtual environment! Consider making/activating one. (See README.md)')

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QSurfaceFormat

from widgets import *

if __name__ == '__main__':    
    qsf = QSurfaceFormat()
    qsf.setRenderableType(QSurfaceFormat.OpenGL)
    qsf.setProfile(QSurfaceFormat.CoreProfile)
    qsf.setVersion(4, 1)
    QSurfaceFormat.setDefaultFormat(qsf)

    app = QApplication(sys.argv)

    imp = IMPWindow()

    sys.exit(app.exec_())
