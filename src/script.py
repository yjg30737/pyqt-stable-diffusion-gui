import gc
import os
import random
import string
import sys

import torch
from PIL import Image


def generate_random_prompt(arr):
    if len(arr) > 0:
        max_len = max(map(lambda x: len(x), arr))
        weights = [i for i in range(max_len, 0, -1)]
        random_prompt = ', '.join(list(filter(lambda x: x != '', [random.choices(_, weights[:len(_)])[0] for _ in arr])))
    else:
        random_prompt = ''
    return random_prompt

def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def get_filename(prompt, cnt, ext, width, height, model_id, suffix=''):
    # replace slash with lowdash from model_id
    model_id = model_id.replace('/', '_')
    return f"{'_'.join(map(lambda x: x.replace(',', ''), prompt.split()[:cnt]))}({width}x{height})-{model_id}_{suffix}{ext}"

def image_to_grid(images, rows, cols):
    assert len(images) == rows * cols

    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def generate_image(pipeline, **args):
    # clear cache to avoid OutOfMemoryError
    gc.collect()
    torch.cuda.empty_cache()

    images = pipeline(**args).images
    return images

def save_image(images, prompt, model_id, ext='.png', save_path='.', suffix=''):
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    filename = ''

    if len(images) > 0:
        img = images[0]
        filename = os.path.join(save_path, get_filename(prompt, 10, ext, width=img.width, height=img.height, model_id=model_id,
                                                        suffix=(suffix if suffix == '' else suffix+'_')+generate_random_string(10)))
        img.save(filename)

    return filename

def open_directory(path):
    if sys.platform.startswith('darwin'):  # macOS
        os.system('open "{}"'.format(path))
    elif sys.platform.startswith('win'):  # Windows
        os.system('start "" "{}"'.format(path))
    elif sys.platform.startswith('linux'):  # Linux
        os.system('xdg-open "{}"'.format(path))
    else:
        print("Unsupported operating system.")