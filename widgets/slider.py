from PyQt5.QtWidgets import QSlider, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class Slider(QVBoxLayout):

    def __init__(self, name, mini=0, maxi=1, default_value=0, direction=Qt.Horizontal, n_steps=None, data_type=int):
        super().__init__()
        self.name = name
        if n_steps is None:
            if data_type == int:
                n_steps = maxi - mini
            else:
                n_steps = 20

        self.data_type = data_type

        self.slider = QSlider(direction)
        self.slider.setMinimum(1)
        self.slider.setMaximum(n_steps)
        
        self._maxi = maxi
        self._mini = mini
        self.set_value(default_value)
        
        self.label = QLabel('')
        self.update_label()
        self.addWidget(self.label)
        self.addWidget(self.slider)
        self.slider.valueChanged.connect(self.update_label)

    def set_value(self, value):
        self.slider.setValue(round(self.slider.minimum() + (value - self._mini) / (self._maxi - self._mini) * (self.slider.maximum() - self.slider.minimum())))

    def value(self):
        value = self._mini + ((self.slider.value() - self.slider.minimum()) / (self.slider.maximum() - self.slider.minimum())) * (self._maxi - self._mini)
        return self.data_type(value)

    def update_label(self):
        if self.data_type == int:
            self.label.setText('{}: {}'.format(self.name, self.value()))
        else:
            self.label.setText('{}: {:.2f}'.format(self.name, self.value()))
