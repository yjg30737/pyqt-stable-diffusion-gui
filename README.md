# pyqt-stable-diffusion-gui
[![](https://dcbadge.vercel.app/api/server/cHekprskVE)](https://discord.gg/cHekprskVE)

PyQt(+PySide) Stable Diffusion GUI

This program allows you to generate AI images based on Stable Diffusion using a PyQt application.

This is the result of my personal testing on how to create a program using PyQt that can smoothly generate AI images.

I kindly ask you to give it a try!

And if you want to give me any feedback or have trouble to install this send me an email or better yet, join the discord channel above :)

## Feature
* Supporting most models' image generation
* Supporting parameters controls
* Being able to choose sampler
* Supporting memory & speed control
* Supporting endless image download
* Supporting LoRA
* No prompt/negative prompt token limit

## Requirements
### PyQt, PySide related 
* PyQt5>=5.14
* PySide6
* qtpy
### Stable Diffusion & HuggingFace related
* diffusers>=0.17.1
* torch
* pillow
* transformers
* accelerate>=0.17.0
* huggingface_hub
* safetensors

## How to Use
1. install cuda
2. install torch from <a href="https://pytorch.org/get-started/locally">here</a>
3. git clone ~
4. pip install -r requirements.txt
5. python main.py
 
## Preview
### GUI
![image](https://github.com/yjg30737/pyqt-stable-diffusion-gui/assets/55078043/f509ab2a-3076-44ad-ae58-faad4fd838d8)
### Image Result
<img src="https://github.com/yjg30737/pyqt-stable-diffusion-gui/assets/55078043/81047351-1a08-46d8-a590-ce22c4f44c0f" width=512 height=512>

<img src="https://github.com/yjg30737/pyqt-stable-diffusion-gui/assets/55078043/4686d60c-1d7b-48fe-ba08-c220bdabf3c0" width=512 height=384>

<img src="https://github.com/yjg30737/pyqt-stable-diffusion-gui/assets/55078043/f11eda53-d1a3-4661-882c-987960890fb1" width=512 height=512>



