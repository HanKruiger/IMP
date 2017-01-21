from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import numpy as np

class DatasetMD:

    def __init__(self, path):
        self.data = np.loadtxt(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
