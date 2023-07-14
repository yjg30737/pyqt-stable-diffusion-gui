import subprocess

import os
import torch
from PyQt5.QtWidgets import QFrame, QLabel, QSpacerItem, QSizePolicy, QTableWidget, QHeaderView, QAbstractItemView
from qtpy.QtCore import Qt, QSettings, Signal
from qtpy.QtWidgets import QLineEdit, QMenu, QAction
from qtpy.QtWidgets import QWidget, QFormLayout, QComboBox, QCheckBox, QGroupBox, QVBoxLayout, \
    QRadioButton, QHBoxLayout, QScrollArea, QPushButton, QFileDialog

from twoColCmbBox import TwoColComboBox


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
    added = Signal(str)

    def __init__(self, default_filename: str = ''):
        super().__init__()
        self.__initVal()
        self.__initUi(default_filename)

    def __initVal(self):
        self.__ext_of_files = ''
        self.__directory = False

    def __initUi(self, default_filename: str = ''):
        self.__pathLineEdit = FindPathLineEdit()
        self.__pathLineEdit.setText(default_filename)

        self.__pathFindBtn = QPushButton('Find...')
        self.__pathFindBtn.clicked.connect(self.__find)
        self.__pathLineEdit.setMaximumHeight(self.__pathFindBtn.sizeHint().height())

        lay = QHBoxLayout()
        lay.addWidget(self.__pathLineEdit)
        lay.addWidget(self.__pathFindBtn)
        lay.setContentsMargins(0, 0, 0, 0)

        self.setLayout(lay)

    def __find(self):
        dirname = QFileDialog.getExistingDirectory(None, 'Open Directory', '', QFileDialog.ShowDirsOnly)
        if dirname:
            self.__pathLineEdit.setText(dirname)
            self.added.emit(dirname)


class SettingsWidget(QScrollArea):
    def __init__(self):
        super(SettingsWidget, self).__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        self.__settings_ini = QSettings('config.ini', QSettings.IniFormat)

        # original_prompt, original_negative_prompt is based on prompt guide
        # https://bootcamp.uxdesign.cc/30-prompts-to-create-a-beautiful-girl-in-stable-diffusion-290dc312b3fc

        # the other's default value, minimum, maximum value of each parameters loosely based on
        # https://platform.stability.ai/docs/features/api-parameters

        # Save configuration values
        if not self.__settings_ini.contains('save_path'):
            self.__settings_ini.setValue("save_path", os.path.expanduser('~'))
        if not self.__settings_ini.contains('torch_dtype'):
            self.__settings_ini.setValue("torch_dtype", 16)
        if not self.__settings_ini.contains('safety_checker'):
            self.__settings_ini.setValue("safety_checker", True)

        if not self.__settings_ini.contains('enable_xformers_memory_efficient_attention'):
            self.__settings_ini.setValue("enable_xformers_memory_efficient_attention", False)
        if not self.__settings_ini.contains('enable_vae_slicing'):
            self.__settings_ini.setValue("enable_vae_slicing", False)
        if not self.__settings_ini.contains('enable_attention_slicing'):
            self.__settings_ini.setValue("enable_attention_slicing", False)
        if not self.__settings_ini.contains('enable_vae_tiling'):
            self.__settings_ini.setValue("enable_vae_tiling", False)
        if not self.__settings_ini.contains('enable_sequential_cpu_offload'):
            self.__settings_ini.setValue("enable_sequential_cpu_offload", False)
        if not self.__settings_ini.contains('enable_model_cpu_offload'):
            self.__settings_ini.setValue("enable_model_cpu_offload", False)

        if not self.__settings_ini.contains('sampler'):
            self.__settings_ini.setValue('sampler', 'DPMSolverMultistepScheduler')

        # Read configuration values
        self.__save_path = self.__settings_ini.value('save_path', type=str)
        self.__torch_dtype = self.__settings_ini.value('torch_dtype', type=str)
        self.__safety_checker = self.__settings_ini.value("safety_checker", type=bool)

        self.__enable_xformers_memory_efficient_attention = self.__settings_ini.value('enable_xformers_memory_efficient_attention', type=bool)
        self.__enable_vae_slicing = self.__settings_ini.value('enable_vae_slicing', type=bool)
        self.__enable_attention_slicing = self.__settings_ini.value('enable_attention_slicing', type=bool)
        self.__enable_vae_tiling = self.__settings_ini.value('enable_vae_tiling', type=bool)
        self.__enable_sequential_cpu_offload = self.__settings_ini.value('enable_sequential_cpu_offload', type=bool)
        self.__enable_model_cpu_offload = self.__settings_ini.value('enable_model_cpu_offload', type=bool)

        self.__sampler = self.__settings_ini.value('sampler', type=str)

    def __initUi(self):
        basicSettingsGrpBox = QGroupBox('General')

        findPathLineEdit = FindPathWidget(self.__save_path)
        findPathLineEdit.added.connect(self.__pathChanged)

        torchDtypeCmbBox = QComboBox()
        torchDtypeCmbBox.addItems(['16', '32'])
        torchDtypeCmbBox.setCurrentText(self.__torch_dtype)
        torchDtypeCmbBox.currentTextChanged.connect(self.__dtypeChanged)

        safetyCheckedChkBox = QCheckBox()
        safetyCheckedChkBox.setChecked(self.__safety_checker)
        safetyCheckedChkBox.toggled.connect(self.__safetyCheckerChanged)

        lay = QFormLayout()
        lay.addRow('Saved Path', findPathLineEdit)
        lay.addRow('Torch DType', torchDtypeCmbBox)
        lay.addRow('Safety Checked', safetyCheckedChkBox)

        basicSettingsGrpBox.setLayout(lay)

        memoryAndSpeedGrpBox = QGroupBox()
        memoryAndSpeedGrpBox.setTitle('Memory and Speed')

        enable_xformers_memory_efficient_attentionChkBox = QCheckBox('enable_xformers_memory_efficient_attention')
        enable_xformers_memory_efficient_attentionChkBox.setChecked(self.__enable_xformers_memory_efficient_attention)
        enable_xformers_memory_efficient_attentionChkBox.toggled.connect(self.__enable_xformers_memory_efficient_attentionChkBoxChanged)

        enable_vae_slicingChkBox = QCheckBox('enable_vae_slicing')
        enable_vae_slicingChkBox.setChecked(self.__enable_vae_slicing)
        enable_vae_slicingChkBox.toggled.connect(self.__enable_vae_slicingChkBoxChanged)

        enable_attention_slicingChkBox = QCheckBox('enable_attention_slicing')
        enable_attention_slicingChkBox.setChecked(self.__enable_attention_slicing)
        enable_attention_slicingChkBox.toggled.connect(self.__enable_attention_slicingChkBoxChanged)

        enable_vae_tilingChkBox = QCheckBox('enable_vae_tiling')
        enable_vae_tilingChkBox.setChecked(self.__enable_vae_tiling)
        enable_vae_tilingChkBox.toggled.connect(self.__enable_vae_tilingChkBoxChanged)

        self.__no_cpu_offloadChkBox = QRadioButton('No CPU offload')
        self.__no_cpu_offloadChkBox.setChecked(not self.__enable_sequential_cpu_offload and not self.__enable_model_cpu_offload)
        self.__no_cpu_offloadChkBox.toggled.connect(self.__no_cpu_offloadChkBoxChanged)

        self.__enable_sequential_cpu_offloadChkBox = QRadioButton('enable_sequential_cpu_offload')
        self.__enable_sequential_cpu_offloadChkBox.setChecked(self.__enable_sequential_cpu_offload)
        self.__enable_sequential_cpu_offloadChkBox.toggled.connect(self.__enable_sequential_cpu_offloadChkBoxChanged)

        self.__enable_model_cpu_offloadChkBox = QRadioButton('enable_model_cpu_offload')
        self.__enable_model_cpu_offloadChkBox.setChecked(self.__enable_model_cpu_offload)
        self.__enable_model_cpu_offloadChkBox.toggled.connect(self.__enable_model_cpu_offloadChkBoxChanged)

        lay = QHBoxLayout()
        lay.addWidget(self.__no_cpu_offloadChkBox)
        lay.addWidget(self.__enable_sequential_cpu_offloadChkBox)
        lay.addWidget(self.__enable_model_cpu_offloadChkBox)

        cpuOffloadGrpBox = QGroupBox('CPU Offload Policy')
        cpuOffloadGrpBox.setLayout(lay)

        lay = QFormLayout()
        lay.addRow('enable_xformers_memory_efficient_attention', enable_xformers_memory_efficient_attentionChkBox)
        lay.addRow('enable_vae_slicing', enable_vae_slicingChkBox)
        lay.addRow('enable_attention_slicing', enable_attention_slicingChkBox)
        lay.addRow('enable_vae_tiling', enable_vae_tilingChkBox)
        lay.addRow(cpuOffloadGrpBox)

        memoryAndSpeedGrpBox.setLayout(lay)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)

        self.__samplerCmbBox = TwoColComboBox([['PNDMScheduler', 'Default scheduler'],
                ['DPMSolverMultistepScheduler', 'the best speed/quality trade-off'],
                ['DPMSolverSinglestepScheduler', ''],
                ['LMSDiscreteScheduler', 'usually leads to better results'],
                ['HeunDiscreteScheduler', ''],
                ['EulerDiscreteScheduler', 'generate high quality results with as little as 30 steps'],
                ['EulerAncestralDiscreteScheduler', 'generate high quality results with as little as 30 steps']])
        self.__samplerCmbBox.setCurrentText(self.__sampler)
        self.__samplerCmbBox.currentTextChanged.connect(self.__samplerChanged)

        lay = QFormLayout()
        lay.addRow('Sampler', self.__samplerCmbBox)

        samplerGrpBox = QGroupBox()
        samplerGrpBox.setTitle('Sampler')
        samplerGrpBox.setLayout(lay)

        self.__addBtn = QPushButton('Add')
        self.__delBtn = QPushButton('Delete')

        self.__addBtn.clicked.connect(self.__addClicked)
        self.__delBtn.clicked.connect(self.__deleteClicked)

        lay = QHBoxLayout()

        lay.addWidget(QLabel('Lora Table'))
        lay.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.MinimumExpanding))
        lay.addWidget(self.__addBtn)
        lay.addWidget(self.__delBtn)
        lay.setContentsMargins(0, 0, 0, 0)

        loraMenuWidget = QWidget()
        loraMenuWidget.setLayout(lay)

        lora_table_header = ['Name', 'Description', 'Link']
        self.__loraTable = QTableWidget()
        self.__loraTable.setColumnCount(len(lora_table_header))
        self.__loraTable.setHorizontalHeaderLabels(lora_table_header)
        self.__loraTable.setRowCount(2)
        self.__loraTable.resizeColumnsToContents()
        self.__loraTable.verticalHeader().setVisible(False)
        self.__loraTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.__loraTable.setSelectionBehavior(QAbstractItemView.SelectRows)

        lay = QVBoxLayout()
        lay.addWidget(loraMenuWidget)
        lay.addWidget(self.__loraTable)

        loraWidget = QWidget()
        loraWidget.setLayout(lay)

        lay = QVBoxLayout()
        lay.addWidget(basicSettingsGrpBox)
        lay.addWidget(memoryAndSpeedGrpBox)
        lay.addWidget(samplerGrpBox)
        lay.addWidget(loraWidget)

        mainWidget = QWidget()
        mainWidget.setLayout(lay)

        self.setWidget(mainWidget)
        self.setWidgetResizable(True)

    def __addClicked(self):
        print('add')

    def __deleteClicked(self):
        print('delete')

    def __pathChanged(self, save_path):
        self.__settings_ini.setValue("save_path", save_path)
        self.__save_path = save_path

    def __dtypeChanged(self, torch_dtype):
        self.__settings_ini.setValue("torch_dtype", torch_dtype)
        self.__torch_dtype = torch_dtype

    def __safetyCheckerChanged(self, f):
        self.__settings_ini.setValue("safety_checker", f)
        self.__safety_checker = f

    def getSavedPath(self):
        return self.__save_path

    def __samplerChanged(self, sampler):
        self.__settings_ini.setValue('sampler', sampler)
        self.__sampler = sampler

    def getTorchDtype(self):
        if self.__torch_dtype == '16':
            return torch.float16
        elif self.__torch_dtype == '32':
            return torch.float32

    def getSafetyChecked(self):
        return self.__safety_checker

    def __enable_xformers_memory_efficient_attentionChkBoxChanged(self, f):
        self.__settings_ini.setValue('enable_xformers_memory_efficient_attention', f)
        self.__enable_xformers_memory_efficient_attention = f

    def __enable_vae_slicingChkBoxChanged(self, f):
        self.__settings_ini.setValue('enable_vae_slicing', f)
        self.__enable_vae_slicing = f

    def __enable_attention_slicingChkBoxChanged(self, f):
        self.__settings_ini.setValue('enable_attention_slicing', f)
        self.__enable_attention_slicing = f

    def __enable_vae_tilingChkBoxChanged(self, f):
        self.__settings_ini.setValue('enable_vae_tiling', f)
        self.__enable_vae_tiling = f

    def __no_cpu_offloadChkBoxChanged(self, f):
        if f:
            self.__enable_sequential_cpu_offloadChkBoxChanged(False)
            self.__enable_model_cpu_offloadChkBoxChanged(False)

    def __enable_sequential_cpu_offloadChkBoxChanged(self, f):
        self.__settings_ini.setValue('enable_sequential_cpu_offload', f)
        self.__enable_sequential_cpu_offload = f

    def __enable_model_cpu_offloadChkBoxChanged(self, f):
        self.__settings_ini.setValue('enable_model_cpu_offload', f)
        self.__enable_model_cpu_offload = f