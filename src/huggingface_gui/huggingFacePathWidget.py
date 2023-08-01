import os
import subprocess

from qtpy.QtCore import Qt, QSettings
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QLineEdit, QMenu, QAction
from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout, QFileDialog, QLabel
from transformers import TRANSFORMERS_CACHE


class FindPathLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.__initUi()

    def __initUi(self):
        self.setMouseTracking(True)
        self.setReadOnly(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.__prepareMenu)

    def mouseMoveEvent(self, e):
        self.__showToolTip()
        return super().mouseMoveEvent(e)

    def __showToolTip(self):
        text = self.text()
        text_width = self.fontMetrics().boundingRect(text).width()

        if text_width > self.width():
            self.setToolTip(text)
        else:
            self.setToolTip('')

    def __prepareMenu(self, pos):
        menu = QMenu(self)
        openDirAction = QAction('Open Path')
        openDirAction.setEnabled(self.text().strip() != '')
        openDirAction.triggered.connect(self.__openPath)
        menu.addAction(openDirAction)
        menu.exec(self.mapToGlobal(pos))

    def __openPath(self):
        filename = self.text()
        path = filename.replace('/', '\\')
        subprocess.Popen(r'explorer /select,"' + path + '"')


class FindPathWidget(QWidget):
    findClicked = Signal()
    onCacheDirSet = Signal(str)

    def __init__(self):
        super().__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        self.__ext_of_files = ''
        self.__directory = False

        self.__settings_struct = QSettings('config.ini', QSettings.IniFormat)
        if self.__settings_struct.contains('CACHE_DIR'):
            pass
        # set the cache_dir if ini file is initially made
        else:
            self.__settings_struct.setValue('CACHE_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        self.onCacheDirSet.emit(self.__settings_struct.value('CACHE_DIR'))

    def __initUi(self):
        self.__pathLineEdit = FindPathLineEdit()
        self.__pathLineEdit.setText(self.__settings_struct.value('CACHE_DIR'))

        self.__pathFindBtn = QPushButton('Find...')

        self.__pathFindBtn.clicked.connect(self.__find)

        self.__pathLineEdit.setMaximumHeight(self.__pathFindBtn.sizeHint().height())

        lay = QHBoxLayout()
        lay.addWidget(self.__pathLineEdit)
        lay.addWidget(self.__pathFindBtn)
        lay.setContentsMargins(0, 0, 0, 0)

        self.setLayout(lay)

    def __customFind(self):
        self.findClicked.emit()

    def __find(self):
        dirname = QFileDialog.getExistingDirectory(None, 'Set Path', '', QFileDialog.ShowDirsOnly)
        if dirname:
            self.__cache_dir = dirname
            self.__handleSettingCacheDir()

    def __handleSettingCacheDir(self):
        self.__pathLineEdit.setText(self.__cache_dir)
        self.__settings_struct.setValue('CACHE_DIR', os.path.normpath(self.__cache_dir))
        self.onCacheDirSet.emit(self.__settings_struct.value('CACHE_DIR'))

    def setLabel(self, text):
        self.layout().insertWidget(0, QLabel(text))

    def getCacheDirectory(self):
        return self.__pathLineEdit.text()

