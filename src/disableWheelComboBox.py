from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QComboBox


class DisableWheelComboBox(QComboBox):
    def __init__(self):
        super(QComboBox, self).__init__()
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        if (event.type() == QEvent.Wheel and
                isinstance(source, QComboBox)):
            return True
        return super().eventFilter(source, event)