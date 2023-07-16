from qtpy.QtCore import Qt, QAbstractTableModel
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QTableView, QHeaderView, QAbstractItemView, QStyledItemDelegate

from disableWheelComboBox import DisableWheelComboBox


class TwoColCmbBoxTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__initUi()

    def __initUi(self):
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setShowGrid(False)


class TwoColTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent):
        return len(self._data)

    def columnCount(self, parent):
        return 2

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]

class TwoColStyleDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.column() == 0:
            option.displayAlignment = Qt.AlignLeft | Qt.AlignVCenter
        elif index.column() == 1:
            option.displayAlignment = Qt.AlignRight | Qt.AlignVCenter
            option.palette.setColor(option.palette.Text, QColor('#888888'))  # Set your desired color here


class TwoColComboBox(DisableWheelComboBox):
    def __init__(self, data):
        super().__init__()
        self.setModel(TwoColTableModel(data))
        view = TwoColCmbBoxTableView(self)

        view.setShowGrid(False)
        view.horizontalHeader().setVisible(False)
        view.verticalHeader().setVisible(False)
        delegate = TwoColStyleDelegate()
        view.setItemDelegate(delegate)
        self.setView(view)