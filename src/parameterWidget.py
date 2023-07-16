from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMainWindow, QApplication, QLabel, QSpinBox, QTextEdit, QWidget, QVBoxLayout, \
    QDoubleSpinBox, QFormLayout, QFrame, QGroupBox, QPushButton, QCheckBox, QScrollArea

class ParameterScrollArea(QScrollArea):
    def __init__(self):
        super(ParameterScrollArea, self).__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        self.__settings_ini = QSettings('config.ini', QSettings.IniFormat)

        # original_prompt, original_negative_prompt is based on prompt guide
        # https://bootcamp.uxdesign.cc/30-prompts-to-create-a-beautiful-girl-in-stable-diffusion-290dc312b3fc

        # the other's default value, minimum, maximum value of each parameters loosely based on
        # https://platform.stability.ai/docs/features/api-parameters

        ### original start ###
        self.__settings_ini.beginGroup('ORIGINAL')
        if not self.__settings_ini.contains('original_prompt'):
            self.__settings_ini.setValue("original_prompt",
                                         "masterpiece, best quality, high resolution, 8K, HDR, beautiful girl, detailed hair, beautiful face, ultra detailed eyes, (hyperdetailed:1.15)")
        if not self.__settings_ini.contains('original_negative_prompt'):
            self.__settings_ini.setValue("original_negative_prompt",
                                         "(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), bad composition, inaccurate eyes, extra digit, fewer digits, (extra arms:1.2)")
        if not self.__settings_ini.contains('original_width'):
            self.__settings_ini.setValue("original_width", 512)
        if not self.__settings_ini.contains('original_height'):
            self.__settings_ini.setValue("original_height", 512)
        if not self.__settings_ini.contains('original_num_inference_steps'):
            self.__settings_ini.setValue("original_num_inference_steps", 20)
        if not self.__settings_ini.contains('original_guidance_scale'):
            self. __settings_ini.setValue("original_guidance_scale", 7.5)
        if not self.__settings_ini.contains('original_rows'):
            self.__settings_ini.setValue("original_rows", 1)
        if not self.__settings_ini.contains('original_cols'):
            self.__settings_ini.setValue("original_cols", 1)

        self.__original_prompt = self.__settings_ini.value('original_prompt', type=str)
        self.__original_negative_prompt = self.__settings_ini.value('original_negative_prompt', type=str)
        self.__original_width = self.__settings_ini.value('original_width', type=int)
        self.__original_height = self.__settings_ini.value('original_height', type=int)
        self.__original_num_inference_steps = self.__settings_ini.value('original_num_inference_steps', type=int)
        self.__original_guidance_scale = self.__settings_ini.value('original_guidance_scale', type=float)
        self.__original_rows = self.__settings_ini.value('original_rows', type=int)
        self.__original_cols = self.__settings_ini.value('original_cols', type=int)
        self.__settings_ini.endGroup()
        ### original end ###

        ### param start ###
        if not self.__settings_ini.contains('prompt'):
            self.__settings_ini.setValue("prompt", self.__original_prompt)
        if not self.__settings_ini.contains('negative_prompt'):
            self.__settings_ini.setValue("negative_prompt", self.__original_negative_prompt)
        if not self.__settings_ini.contains('width'):
            self.__settings_ini.setValue("width", self.__original_width)
        if not self.__settings_ini.contains('height'):
            self.__settings_ini.setValue("height", self.__original_height)
        if not self.__settings_ini.contains('num_inference_steps'):
            self.__settings_ini.setValue("num_inference_steps", self.__original_num_inference_steps)
        if not self.__settings_ini.contains('guidance_scale'):
            self.__settings_ini.setValue("guidance_scale", self.__original_guidance_scale)
        if not self.__settings_ini.contains('rows'):
            self.__settings_ini.setValue("rows", self.__original_rows)
        if not self.__settings_ini.contains('cols'):
            self.__settings_ini.setValue("cols", self.__original_cols)

        self.__prompt = self.__settings_ini.value("prompt", type=str)
        self.__negative_prompt = self.__settings_ini.value("negative_prompt", type=str)
        self.__width = self.__settings_ini.value("width", type=int)
        self.__height = self.__settings_ini.value("height", type=int)
        self.__num_inference_steps = self.__settings_ini.value("num_inference_steps", type=int)
        self.__guidance_scale = self.__settings_ini.value("guidance_scale", type=float)
        self.__rows = self.__settings_ini.value("rows", type=int)
        self.__cols = self.__settings_ini.value("cols", type=int)
        ### param end ###

    def __initUi(self):
        self.setWindowTitle('Stable Diffusion GUI')

        self.__promptLbl = QLabel("Prompt:")
        self.__promptTextEdit = QTextEdit()
        self.__promptTextEdit.setPlaceholderText(self.__original_prompt)
        self.__promptTextEdit.setPlainText(self.__prompt)
        self.__promptTextEdit.textChanged.connect(self.__onPromptChanged)
        self.__promptTextEdit.setAcceptRichText(False)

        self.__negativePromptLbl = QLabel("Negative Prompt:")
        self.__negativePromptTextEdit = QTextEdit()
        self.__negativePromptTextEdit.setPlaceholderText(self.__original_negative_prompt)
        self.__negativePromptTextEdit.setPlainText(self.__negative_prompt)
        self.__negativePromptTextEdit.textChanged.connect(self.__onNegativePromptChanged)
        self.__negativePromptTextEdit.setAcceptRichText(False)

        randomPromptCheckBox = QCheckBox('Insert random text each time an image is generated at the end of the prompt (Not Available now)')
        randomPromptCheckBox.setDisabled(True)

        lay = QVBoxLayout()
        lay.addWidget(self.__promptLbl)
        lay.addWidget(self.__promptTextEdit)
        lay.addWidget(self.__negativePromptLbl)
        lay.addWidget(self.__negativePromptTextEdit)
        lay.addWidget(randomPromptCheckBox)

        promptWidget = QGroupBox()
        promptWidget.setTitle('Prompt')
        promptWidget.setLayout(lay)

        self.__widthSpinBox = QSpinBox()
        self.__widthSpinBox.setRange(128, 1024)
        self.__widthSpinBox.setValue(self.__width)
        self.__widthSpinBox.setSingleStep(8)
        self.__widthSpinBox.valueChanged.connect(self.__widthSpinBoxValueChanged)

        self.__heightSpinBox = QSpinBox()
        self.__heightSpinBox.setRange(128, 1024)
        self.__heightSpinBox.setValue(self.__height)
        self.__heightSpinBox.setSingleStep(8)
        self.__heightSpinBox.valueChanged.connect(self.__heightSpinBoxValueChanged)

        self.__inferenceStepsSpinBox = QSpinBox()
        self.__inferenceStepsSpinBox.setRange(20, 50)
        self.__inferenceStepsSpinBox.setValue(self.__num_inference_steps)
        self.__inferenceStepsSpinBox.valueChanged.connect(self.__inferenceStepsSpinBoxValueChanged)

        self.__guidanceScaleSpinBox = QDoubleSpinBox()
        self.__guidanceScaleSpinBox.setRange(4, 14)
        self.__guidanceScaleSpinBox.setValue(self.__guidance_scale)
        self.__guidanceScaleSpinBox.valueChanged.connect(self.__guidanceScaleSpinBoxValueChanged)

        self.__rowSpinBox = QSpinBox()
        self.__rowSpinBox.setMinimum(1)
        self.__rowSpinBox.setValue(self.__rows)
        self.__rowSpinBox.valueChanged.connect(self.__rowSpinBoxValueChanged)

        self.__columnSpinBox = QSpinBox()
        self.__columnSpinBox.setMinimum(1)
        self.__columnSpinBox.setValue(self.__cols)
        self.__columnSpinBox.valueChanged.connect(self.__columnSpinBoxValueChanged)

        lay = QFormLayout()
        lay.addRow('Width', self.__widthSpinBox)
        lay.addRow('Height', self.__heightSpinBox)
        lay.addRow('Inference Steps', self.__inferenceStepsSpinBox)
        lay.addRow('Guidance Scale', self.__guidanceScaleSpinBox)
        lay.addRow('Rows', self.__rowSpinBox)
        lay.addRow('Columns', self.__columnSpinBox)

        etcParamsWidget = QGroupBox()
        etcParamsWidget.setTitle('etc')
        etcParamsWidget.setLayout(lay)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)

        resetBtn = QPushButton('Reset')
        resetBtn.clicked.connect(self.__reset)

        lay = QVBoxLayout()
        lay.addWidget(promptWidget)
        lay.addWidget(etcParamsWidget)
        lay.addWidget(resetBtn)

        mainWidget = QWidget()
        mainWidget.setLayout(lay)

        self.setWidget(mainWidget)
        self.setWidgetResizable(True)

    def __onPromptChanged(self):
        self.__prompt = self.__promptTextEdit.toPlainText()
        self.__settings_ini.setValue('prompt', self.__prompt)

    def __onNegativePromptChanged(self):
        self.__negative_prompt = self.__negativePromptTextEdit.toPlainText()
        self.__settings_ini.setValue('negative_prompt', self.__negative_prompt)

    def __widthSpinBoxValueChanged(self, v):
        self.__width = v
        self.__settings_ini.setValue('width', self.__width)
    def __heightSpinBoxValueChanged(self, v):
        self.__height = v
        self.__settings_ini.setValue('height', self.__height)
    def __inferenceStepsSpinBoxValueChanged(self, v):
        self.__inferenceSteps = v
        self.__settings_ini.setValue('num_inference_steps', self.__num_inference_steps)
    def __guidanceScaleSpinBoxValueChanged(self, v):
        self.__guidanceScale = v
        self.__settings_ini.setValue('guidance_scale', self.__guidance_scale)
    def __rowSpinBoxValueChanged(self, v):
        self.__rows = v
        self.__settings_ini.setValue('rows', self.__rows)
    def __columnSpinBoxValueChanged(self, v):
        self.__cols = v
        self.__settings_ini.setValue('cols', self.__cols)

    def __reset(self):
        self.__promptTextEdit.setText(self.__original_prompt)
        self.__negativePromptTextEdit.setText(self.__negative_prompt)
        self.__widthSpinBox.setValue(self.__original_width)
        self.__heightSpinBox.setValue(self.__original_height)
        self.__inferenceStepsSpinBox.setValue(self.__original_num_inference_steps)
        self.__guidanceScaleSpinBox.setValue(self.__original_guidance_scale)
        self.__rowSpinBox.setValue(self.__original_rows)
        self.__columnSpinBox.setValue(self.__original_cols)