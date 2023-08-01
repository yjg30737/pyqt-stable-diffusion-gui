import os
import sys

from .huggingFacePathWidget import FindPathWidget

from qtpy.QtCore import Qt, Signal, QSettings
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QSpacerItem, QSizePolicy, \
    QPushButton, QDialog, QMessageBox

from .huggingFaceModelClass import HuggingFaceModelClass
from .huggingFaceModelInputDialog import HuggingFaceModelInputDialog
from .huggingFaceModelTableWidget import HuggingFaceModelTableWidget

QApplication.setWindowIcon(QIcon('hf-logo.svg'))


class HuggingFaceModelWidget(QWidget):
    onModelAdded = Signal(str)
    onModelDeleted = Signal(str)
    onModelSelected = Signal(str)
    onCacheDirSet = Signal(str)

    def __init__(self, certain_models=None, parent=None):
        super(HuggingFaceModelWidget, self).__init__(parent)
        self.__initVal(certain_models)
        self.__initUi()

    def __initVal(self, certain_models):
        self.__certain_models = certain_models

    def __initUi(self):
        self.setWindowTitle('HuggingFace Model Table')

        self.__findPathWidget = FindPathWidget()
        self.__cache_dir = self.__findPathWidget.getCacheDirectory()
        self.__findPathWidget.onCacheDirSet.connect(self.setCacheDir)

        self.__addBtn = QPushButton('Add')
        self.__delBtn = QPushButton('Delete')

        self.__addBtn.clicked.connect(self.__addClicked)
        self.__delBtn.clicked.connect(self.__deleteClicked)

        lay = QHBoxLayout()

        lay.addWidget(QLabel('Model Table'))
        lay.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.MinimumExpanding))
        lay.addWidget(QLabel('Cache Directory'))
        lay.addWidget(self.__findPathWidget)
        lay.addWidget(self.__addBtn)
        lay.addWidget(self.__delBtn)
        lay.setContentsMargins(0, 0, 0, 0)

        menuWidget = QWidget()
        menuWidget.setLayout(lay)

        self.__hf_class = HuggingFaceModelClass()
        self.__hf_class.setCacheDir(self.__cache_dir)
        self.__hf_class.setText2ImageOnly(True)

        self.__modelTableWidget = HuggingFaceModelTableWidget()
        self.__modelTableWidget.setHorizontalHeaderLabels(['Name', 'Visit'])
        self.__modelTableWidget.currentCellChanged.connect(self.__currentCellChanged)

        self.__totalSizeLbl = QLabel()
        self.__totalSizeLbl.setAlignment(Qt.AlignRight)

        self.setCacheDir(self.__cache_dir)

        lay = QVBoxLayout()
        lay.addWidget(menuWidget)
        lay.addWidget(self.__modelTableWidget)
        lay.addWidget(self.__totalSizeLbl)

        self.setLayout(lay)

        self.resize(640, 300)

    def __addClicked(self):
        dialog = HuggingFaceModelInputDialog(self.__hf_class)
        reply = dialog.exec()
        if reply == QDialog.Accepted:
            try:
                model = dialog.getModel()
                # add model in the table
                self.__modelTableWidget.addModels([model])
                self.onModelAdded.emit(model['id'])
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def __deleteClicked(self):
        model_name = self.__modelTableWidget.getCurrentRowModelName()
        self.__hf_class.removeHuggingFaceModel(model_name)
        cur_row = self.__modelTableWidget.currentRow()
        self.__modelTableWidget.removeRow(cur_row)
        self.__modelTableWidget.setCurrentCell(max(0, min(cur_row, self.__modelTableWidget.rowCount()-1)), 0)
        self.__delBtn.setEnabled(self.__modelTableWidget.rowCount() != 0)
        self.onModelDeleted.emit(model_name)

    def __currentCellChanged(self, currentRow, currentColumn, previousRow, previousColumn):
        cur_item = self.__modelTableWidget.item(currentRow, 0)
        if cur_item:
            self.__delBtn.setEnabled(True)
            cur_model_name = cur_item.text()
            self.onModelSelected.emit(cur_model_name)
        else:
            self.__delBtn.setEnabled(False)

    def setCacheDir(self, cache_dir):
        # make directory tree if it is not existed
        os.makedirs(cache_dir, exist_ok=True)

        self.__cache_dir = cache_dir
        self.__modelTableWidget.clearContents()
        self.__modelTableWidget.setRowCount(0)
        self.__hf_class.setCacheDir(self.__cache_dir)
        models = self.__hf_class.getModels(self.__certain_models)
        if len(models) == 0:
            self.__delBtn.setEnabled(False)
        self.__modelTableWidget.addModels(models)
        self.onCacheDirSet.emit(cache_dir)

    def getCurrentModelName(self):
        cur_item = self.__modelTableWidget.item(self.__modelTableWidget.currentRow(), 0)
        cur_model_name = ''
        if cur_item:
            cur_model_name = cur_item.text()
        return cur_model_name

    def getCurrentModelObject(self):
        """
        after called this, you should use from_pretrained
        :return:
        """
        return self.__hf_class.getModelObject(self.getCurrentModelName())

    def getCertainModelObject(self, model_name):
        """
        after called this, you should use from_pretrained
        :return:
        """
        return self.__hf_class.getModelObject(model_name)

    def getModelTable(self):
        return self.__modelTableWidget

    def selectCurrentModel(self, model_name):
        """
        select the row which has "model_name" as an Name
        """
        items = self.__modelTableWidget.findItems(model_name, Qt.MatchExactly)
        if len(items) > 0:
            self.__modelTableWidget.setCurrentCell(items[0].row(), 0)

    def getModelClass(self):
        return self.__hf_class