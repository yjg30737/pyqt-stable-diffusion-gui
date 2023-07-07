from typing import Union

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QDialog, QVBoxLayout, QFrame, QPushButton, QHBoxLayout, QWidget, QApplication, QLabel, \
    QSpacerItem, QSizePolicy, QMessageBox

from .huggingFaceModelClass import HuggingFaceModelClass
from .huggingFaceModelInstallWidget import HuggingFaceModelInstallWidget


class HuggingFaceModelInputDialog(QDialog):
    def __init__(self, hf_class, parent=None):
        super().__init__(parent)
        self.__initVal(hf_class)
        self.__initUi()

    def __initVal(self, hf_class):
        self.__hf_class = hf_class
        self.__model = ''

    def __initUi(self):
        self.setWindowTitle('New...')
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)

        self.__huggingFaceModelLoadingWidget = HuggingFaceModelInstallWidget(self.__hf_class)
        self.__huggingFaceModelLoadingWidget.onInstalled.connect(self.__setAccept)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)

        self.__criticalLbl = QLabel()
        self.__criticalLbl.setStyleSheet("color: {}".format(QColor(255, 0, 0).name()))

        cancelBtn = QPushButton('Cancel')
        cancelBtn.clicked.connect(self.close)

        lay = QHBoxLayout()
        lay.addWidget(self.__criticalLbl)
        lay.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.MinimumExpanding))
        lay.addWidget(cancelBtn)
        lay.setAlignment(Qt.AlignRight)
        lay.setContentsMargins(0, 0, 0, 0)

        bottomWidget = QWidget()
        bottomWidget.setLayout(lay)

        lay = QVBoxLayout()
        lay.addWidget(self.__huggingFaceModelLoadingWidget)
        lay.addWidget(sep)
        lay.addWidget(bottomWidget)

        self.setLayout(lay)

    def getModel(self):
        return self.__model

    def __setAccept(self, model: dict):
        self.__model = model
        self.accept()