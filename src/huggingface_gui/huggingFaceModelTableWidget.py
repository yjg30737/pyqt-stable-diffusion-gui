import typing

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QTableWidget, QHeaderView, QAbstractItemView, QTableWidgetItem, QLabel


class HuggingFaceModelTableWidget(QTableWidget):
    def __init__(self):
        super(HuggingFaceModelTableWidget, self).__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        self.__header_labels = {v: i for (i, v) in enumerate(['Name', 'Visit'])}

    def __initUi(self):
        self.setHorizontalHeaderLabels(self.__header_labels.keys())
        self.resizeColumnsToContents()
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

    def open_link(self, url):
        QDesktopServices.openUrl(QUrl(url))

    def set_hyperlink(self, row, column, text):
        label = QLabel(text)
        label.setOpenExternalLinks(True)
        label.setAlignment(Qt.AlignCenter)
        self.setCellWidget(row, column, label)

    def addModels(self, models: list):
        for i in range(len(models)):
            cur_table_idx = self.rowCount()

            self.setRowCount(cur_table_idx+1)
            model = models[i]

            model_id = model['id'] if isinstance(model, dict) else model

            # id
            if self.__header_labels.get('Name', '') != '':
                model_id_item = QTableWidgetItem(model_id)
                model_id_item.setTextAlignment(Qt.AlignCenter)
                self.setItem(cur_table_idx, self.__header_labels['Name'], model_id_item)

            # visit
            if self.__header_labels.get('Visit', '') != '':
                hyperlink_tag = f'<a href="https://huggingface.co/{model_id}">Link</a>'
                self.set_hyperlink(cur_table_idx, self.__header_labels['Visit'], hyperlink_tag)

            self.setCurrentCell(0, 0)

        self.resizeColumnsToContents()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sortByColumn(0, Qt.AscendingOrder)

    def setHorizontalHeaderLabels(self, labels: typing.Iterable[str]) -> None:
        self.__header_labels = {v: i for (i, v) in enumerate(labels)}
        self.setColumnCount(len(self.__header_labels))
        super(HuggingFaceModelTableWidget, self).setHorizontalHeaderLabels(self.__header_labels.keys())

    def getCurrentRowModelName(self):
        return self.item(self.currentRow(), 0).text()