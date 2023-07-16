from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

setup(
    name='pyqt-stable-diffusion-gui',
    version='0.0.124',
    author='Jung Gyu Yoon',
    author_email='yjg30737@gmail.com',
    license='MIT',
    packages=find_packages(),
    package_data={'src': ['hf-logo.svg'], 'src.ico': ['close.svg', 'prompt.svg', 'setting.svg', 'sidebar.svg', 'table.svg']},
    description='PyQt Stable Diffusion GUI',
    url='https://github.com/yjg30737/pyqt-stable-diffusion-gui.git',
    long_description_content_type='text/markdown',
    long_description=long_description,
    install_requires=[
        'PyQt5>=5.14',
        'PySide6',
        'qtpy',
        'diffusers>=0.17.1',
        'torch',
        'pillow',
        'transformers',
        'accelerate>=0.17.0',
        'huggingface_hub',
        'safetensors'
    ]
)