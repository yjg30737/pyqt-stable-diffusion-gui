import os
import sys

from svgButton import SvgButton
from transformers import TRANSFORMERS_CACHE

from settingsWidget import SettingsWidget

# Get the absolute path of the current script file
script_path = os.path.abspath(__file__)

# Get the root directory by going up one level from the script directory
project_root = os.path.dirname(os.path.dirname(script_path))

sys.path.insert(0, project_root)
sys.path.insert(0, os.getcwd())  # Add the current directory as well

from qtpy.QtCore import QCoreApplication, Qt, QSettings
from qtpy.QtGui import QGuiApplication, QFont, QIcon

from huggingface_gui.huggingFaceModelWidget import HuggingFaceModelWidget
from parameterWidget import ParameterScrollArea
from script import generate_random_prompt, open_directory
from stableDiffusionClass import StableDiffusionWrapper
from thread import Thread

from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, \
    QPushButton, QAction, QMenu, QWidgetAction, QLabel, QToolBar, QSplitter, QScrollArea, QHBoxLayout, QSpinBox, \
    QCheckBox, QMessageBox

# for testing pyside6
# if you use pyside6 already, you don't have to remove the #
# os.environ['QT_API'] = 'pyside6'

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # HighDPI support
# qt version should be above 5.14
# todo check the qt version with qtpy
if os.environ['QT_API'] == 'pyqt5' or os.environ['QT_API'] != 'pyside6':
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

QApplication.setFont(QFont('Arial', 12))
QApplication.setWindowIcon(QIcon('hf-logo.svg'))


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        self.__stable_diffusion_wrapper = StableDiffusionWrapper()
        self.__current_model_lbl_prefix = 'Current Model:'
        self.__settings_ini = QSettings('config.ini', QSettings.IniFormat)

        if not self.__settings_ini.contains('generation_count'):
            self.__settings_ini.setValue("generation_count", 1)
        if not self.__settings_ini.contains('is_infinite'):
            self.__settings_ini.setValue("is_infinite", False)

        self.__generation_count = self.__settings_ini.value("generation_count", type=int)
        self.__is_infinite = self.__settings_ini.value("is_infinite", type=bool)
        self.__current_model = self.__settings_ini.value('current_model', type=str)

    def __initUi(self):
        self.setWindowTitle('Stable Diffusion GUI')

        self.__currentModelLbl = QLabel()
        self.__generateBtn = QPushButton('Generate')

        ### huggingface model widget start ###
        self.__huggingFaceModelWidget = HuggingFaceModelWidget()
        self.__huggingFaceModelWidget.onModelAdded.connect(self.__onModelAdded)
        self.__huggingFaceModelWidget.onModelDeleted.connect(self.__onModelDeleted)
        self.__huggingFaceModelWidget.onModelSelected.connect(self.__onModelSelected)
        self.__huggingFaceModelWidget.onCacheDirSet.connect(self.__onCacheDirSet)
        # if current model is empty string
        if not self.__current_model:
            self.__current_model = self.__huggingFaceModelWidget.getCurrentModelName()
        # init model here
        if not self.__settings_ini.contains('current_model'):
            self.__settings_ini.setValue('current_model', self.__current_model)
        self.__huggingFaceModelWidget.selectCurrentModel(self.__current_model)

        self.__huggingFaceModelWidgetScrollArea = QScrollArea()
        self.__huggingFaceModelWidgetScrollArea.setWidget(self.__huggingFaceModelWidget)
        self.__huggingFaceModelWidgetScrollArea.setWidgetResizable(True)

        self.__paramScrollArea = ParameterScrollArea()

        self.__currentModelLbl.setText(f'{self.__current_model_lbl_prefix} {self.__current_model}')
        self.__currentModelLbl.setContentsMargins(5, 5, 5, 5)
        self.__currentModelLbl.setAlignment(Qt.AlignCenter)
        self.__currentModelLbl.setStyleSheet('QLabel { background-color: #CCC;  border: 1px solid #888 }')
        self.__currentModelLbl.setMaximumHeight(self.__currentModelLbl.sizeHint().height())

        lbl = QLabel('Model List')
        lbl.setContentsMargins(2, 2, 2, 2)
        lbl.setStyleSheet('QLabel { background-color: #BBB; border: 1px solid gray }')

        lay = QVBoxLayout()
        lay.addWidget(lbl)
        lay.addWidget(self.__huggingFaceModelWidgetScrollArea)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.__huggingFaceModelWidgetScrollAreaOuterWidget = QWidget()
        self.__huggingFaceModelWidgetScrollAreaOuterWidget.setLayout(lay)
        ### huggingface model widget end ###

        ### setting widget start ###
        self.__settingsWidget = SettingsWidget()

        lbl = QLabel('Settings')
        lbl.setContentsMargins(2, 2, 2, 2)
        lbl.setStyleSheet('QLabel { background-color: #BBB; border: 1px solid gray }')
        lbl.setMaximumHeight(lbl.sizeHint().height())

        lay = QVBoxLayout()
        lay.addWidget(lbl)
        lay.addWidget(self.__settingsWidget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.__settingsOuterWidget = QWidget()
        self.__settingsOuterWidget.setLayout(lay)
        ### setting widget end ###

        leftWidget = QSplitter()
        leftWidget.setOrientation(Qt.Vertical)
        leftWidget.addWidget(self.__huggingFaceModelWidgetScrollAreaOuterWidget)
        leftWidget.addWidget(self.__settingsOuterWidget)
        leftWidget.setChildrenCollapsible(False)
        leftWidget.setHandleWidth(2)
        leftWidget.setStyleSheet(
        '''
        QSplitter::handle:vertical
        {
            background: #CCC;
            height: 1px;
        }
        ''')

        ### prompt widget start ###
        lbl = QLabel('Parameter')
        lbl.setContentsMargins(2, 2, 2, 2)
        lbl.setStyleSheet('QLabel { background-color: #BBB; border: 1px solid gray }')

        lay = QVBoxLayout()
        lay.addWidget(lbl)
        lay.addWidget(self.__paramScrollArea)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.__promptScrollAreaWidget = QWidget()
        self.__promptScrollAreaWidget.setLayout(lay)
        ### prompt widget end ###

        ### generate widget start ###
        self.__generationCountSpinBox = QSpinBox()
        self.__generationCountSpinBox.setRange(1, 100000)
        self.__generationCountSpinBox.setValue(self.__generation_count)
        self.__generationCountSpinBox.setAccelerated(True)
        self.__generationCountSpinBox.valueChanged.connect(self.__generationCountSpinBoxChanged)

        self.__infiniteChkBox = QCheckBox('Keep Generating')
        self.__infiniteChkBox.setChecked(self.__is_infinite)
        self.__infiniteChkBox.toggled.connect(self.__infiniteChkBoxStateChanged)

        self.__generateBtn.clicked.connect(self.__generate)
        self.__generateBtn.setEnabled(self.__huggingFaceModelWidget.getModelTable().rowCount() > 0)

        lay = QHBoxLayout()
        lay.addWidget(self.__generationCountSpinBox)
        lay.addWidget(self.__infiniteChkBox)
        lay.addWidget(self.__generateBtn)

        self.__generateWidget = QWidget()
        self.__generateWidget.setLayout(lay)
        self.__generateWidget.setMaximumHeight(self.__generateWidget.sizeHint().height())
        ### generate widget end ###

        ### main widget ###
        mainSplitter = QSplitter()
        mainSplitter.addWidget(leftWidget)
        mainSplitter.addWidget(self.__promptScrollAreaWidget)
        mainSplitter.setChildrenCollapsible(False)
        mainSplitter.setHandleWidth(2)
        mainSplitter.setStyleSheet(
        '''
        QSplitter::handle:horizontal
        {
            background: #CCC;
            height: 1px;
        }
        ''')

        lay = QVBoxLayout()
        lay.addWidget(self.__currentModelLbl)
        lay.addWidget(mainSplitter)
        lay.addWidget(self.__generateWidget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        mainWidget = QWidget()
        mainWidget.setLayout(lay)

        self.setCentralWidget(mainWidget)
        ### main widget ###

        self.__setActions()
        self.__setMenuBar()
        self.__setToolBar()

    def __setActions(self):
        # menu action
        self.__exitAction = QAction("Exit", self)
        self.__exitAction.triggered.connect(self.close)

        self.__modelTableAction = QWidgetAction(self)
        self.__modelTableBtn = SvgButton()
        self.__modelTableBtn.setIcon('ico/table.svg')
        self.__modelTableBtn.setToolTip('Model Table')
        self.__modelTableBtn.setCheckable(True)
        self.__modelTableBtn.setChecked(True)
        self.__modelTableBtn.toggled.connect(self.__huggingFaceModelWidgetScrollAreaOuterWidget.setVisible)
        self.__modelTableAction.setDefaultWidget(self.__modelTableBtn)

        self.__promptAction = QWidgetAction(self)
        self.__promptBtn = SvgButton()
        self.__promptBtn.setIcon('ico/prompt.svg')
        self.__promptBtn.setToolTip('Prompt Generator')
        self.__promptBtn.setCheckable(True)
        self.__promptBtn.setChecked(True)
        self.__promptBtn.toggled.connect(self.__promptScrollAreaWidget.setVisible)
        self.__promptAction.setDefaultWidget(self.__promptBtn)

        self.__settingsAction = QWidgetAction(self)
        self.__settingBtn = SvgButton()
        self.__settingBtn.setIcon('ico/setting.svg')
        self.__settingBtn.setToolTip('Settings')
        self.__settingBtn.setCheckable(True)
        self.__settingBtn.setChecked(True)
        self.__settingBtn.toggled.connect(self.__settingsOuterWidget.setVisible)
        self.__settingsAction.setDefaultWidget(self.__settingBtn)

    def __setMenuBar(self):
        menubar = self.menuBar()

        # create the "File" menu
        fileMenu = QMenu("File", self)
        fileMenu.addAction(self.__exitAction)
        menubar.addMenu(fileMenu)

    def __setToolBar(self):
        toolbar = QToolBar()
        toolbar.layout().setSpacing(2)
        toolbar.addAction(self.__modelTableAction)
        toolbar.addAction(self.__settingsAction)
        toolbar.addAction(self.__promptAction)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

    def __generationCountSpinBoxChanged(self, v):
        self.__generation_count = v
        self.__settings_ini.setValue('generation_count', self.__generation_count)

    def __infiniteChkBoxStateChanged(self, v):
        self.__is_infinite = v
        self.__generationCountSpinBox.setEnabled(not self.__is_infinite)
        self.__settings_ini.setValue('is_infinite', self.__is_infinite)

    def __generate(self):
        try:
            # get parameters which are necessary to initialize the pipeline
            cache_dir = self.__settings_ini.value('CACHE_DIR')
            torch_dtype = self.__settingsWidget.getTorchDtype()
            safety_checker = self.__settingsWidget.getSafetyChecked()

            enable_xformers_memory_efficient_attention = self.__settings_ini.value('enable_xformers_memory_efficient_attention', type=bool)
            enable_vae_slicing = self.__settings_ini.value('enable_vae_slicing', type=bool)
            enable_attention_slicing = self.__settings_ini.value('enable_attention_slicing', type=bool)
            enable_vae_tiling = self.__settings_ini.value('enable_vae_tiling', type=bool)
            enable_sequential_cpu_offload = self.__settings_ini.value('enable_sequential_cpu_offload', type=bool)
            enable_model_cpu_offload = self.__settings_ini.value('enable_model_cpu_offload', type=bool)

            sampler = self.__settings_ini.value('sampler', type=str)

            self.__stable_diffusion_wrapper.init_wrapper(self.__current_model, cache_dir, torch_dtype, safety_checker, sampler)

            lora_paths = self.__settingsWidget.getLoraPaths()
            for lora_path in lora_paths:
                self.__stable_diffusion_wrapper.load_lora_weights(lora_path)

            width = self.__settings_ini.value('width', type=int)
            height = self.__settings_ini.value('height', type=int)
            prompt = self.__settings_ini.value('prompt', type=str)
            negative_prompt = self.__settings_ini.value('negative_prompt', type=str)
            num_inference_steps = self.__settings_ini.value('num_inference_steps', type=int)
            guidance_scale = self.__settings_ini.value('guidance_scale', type=float)
            rows = self.__settings_ini.value('rows', type=int)
            cols = self.__settings_ini.value('cols', type=int)

            pipeline_args = {
                'width': width,
                'height': height,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'num_images_per_prompt': rows*cols
            }

            save_path = self.__settingsWidget.getSavedPath()

            self.__stable_diffusion_wrapper.set_saving_memory_attr(
                enable_xformers_memory_efficient_attention,
                enable_vae_slicing,
                enable_attention_slicing,
                enable_vae_tiling,
                enable_sequential_cpu_offload,
                enable_model_cpu_offload
            )

            pipeline_args = self.__stable_diffusion_wrapper.forward_embeddings_through_text_encoder(pipeline_args)

            pipeline = self.__stable_diffusion_wrapper.get_pipeline()

            generation_count = -1 if self.__is_infinite else self.__generation_count

            self.__t = Thread(pipeline=pipeline, generation_count=generation_count, model_id=self.__current_model, prompt_text_for_filename=prompt, save_path=save_path, rows=rows, cols=cols, **pipeline_args)
            self.__t.started.connect(self.__started)
            self.__t.finished.connect(self.__t.deleteLater)
            self.__t.generateFinished.connect(self.__generateFinished)
            self.__t.generateFailed.connect(self.__generateFailed)
            self.__t.start()
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", str(e))

    def __toggleWidgetByRunning(self, f: bool):
        self.__huggingFaceModelWidgetScrollAreaOuterWidget.setEnabled(f)
        self.__settingsOuterWidget.setEnabled(f)
        self.__promptScrollAreaWidget.setEnabled(f)
        self.__generateWidget.setEnabled(f)

    def __started(self):
        self.__toggleWidgetByRunning(False)

    def __generateFinished(self, filenames: list):
        print('\n'.join(filenames))
        self.__toggleWidgetByRunning(True)
        save_path = self.__settingsWidget.getSavedPath()
        open_directory(save_path)

    def __generateFailed(self, e):
        QMessageBox.critical(self, "Error", str(e))
        self.__toggleWidgetByRunning(True)

    def __onModelSelected(self, model):
        self.__current_model = model
        self.__currentModelLbl.setText(f'{self.__current_model_lbl_prefix} {self.__current_model}')
        self.__settings_ini.setValue('current_model', self.__current_model)
        self.__generateBtn.setEnabled(self.__huggingFaceModelWidget.getModelTable().rowCount() > 0)

    def __onModelAdded(self, model):
        pass

    def __onModelDeleted(self, model):
        if self.__huggingFaceModelWidget.getModelTable().rowCount() == 0:
            self.__onModelSelected('')
        self.__generateBtn.setEnabled(self.__huggingFaceModelWidget.getModelTable().rowCount() > 0)

    def __onCacheDirSet(self, cache_dir):
        self.__generateBtn.setEnabled(self.__huggingFaceModelWidget.getModelTable().rowCount() > 0)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1024, 768)
    w.show()
    sys.exit(app.exec())