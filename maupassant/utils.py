import os
import json
import time
import pickle
import datetime
import functools

import numpy as np

import tensorflow as tf

from maupassant.settings import MODEL_PATH


def text_format(txt_color='white', txt_style='normal', bg_color=None, end=False):
    color = {
        'white': 0, 'red': 1, "green": 2, "yellow": 3, "blue": 4,
        "purple": 5, "cyan": 6, "black": 7
    }
    style = {'normal': 0, 'bold': 1, "underline": 2}
    if end:
        return "\033[0m"

    if not bg_color:
        return f" \x1b[{str(style[txt_style])};3{str(color[txt_color])}m "

    return f" \033[{str(style[txt_style])};4{str(color[bg_color])};3{str(color[txt_color])}m "


def mark_format(mark_color='purple', close=False):
    map_color = {
        "purple": "aa9cfc", "blue": "7aecec", "kaki": "bfe1d9",
        "orange": "feca74", "green": "bfeeb7", "magenta": "aa9cfc"
    }
    color = map_color[mark_color]
    if close:
        return ' </mark> '

    return f' <mark class="entity" style="background:#{color};padding:0.45em;line-height:1;border-radius:0.35em"> '


def timer(func):

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        run_time = time.perf_counter() - start_time
        bg_fw = text_format(txt_color='white', bg_color='green', txt_style='bold')
        fc = text_format(txt_color='cyan', txt_style='bold')
        fb = text_format(txt_color='blue', txt_style='bold')
        end = text_format(end=True)

        print(f"{bg_fw}Function:{end}{fc}{func.__name__}{end}")
        print(f"{bg_fw}kwargs:{end}{fb}{kwargs}{end}")
        print(f"{bg_fw}Duration:{end}{fb}{run_time*1000:.3f}ms{end}")
        return value

    return wrapper_timer


def predict_format(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(kwargs['x'], str):
            kwargs['x'] = np.asarray([kwargs['x']])
        if isinstance(kwargs['x'], list):
            kwargs['x'] = np.asarray(kwargs['x'])

        return func(*args, **kwargs)

    return wrapper


class GeneratePath(object):

    def __init__(self, base_path=MODEL_PATH, name="classifier"):
        self.base_path = base_path
        self.name = name
        self.date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.paths = self.define_paths()

    def define_paths(self):
        base_dir = os.path.join(self.base_path, f"{self.date}_{self.name}")

        return {
            "path": base_dir,
            "model_path": os.path.join(base_dir, 'model'),
            "model_plot_path":  os.path.join(base_dir, "model.jpg"),
            "model_info_path": os.path.join(base_dir, "model.json"),
            "metrics_path": os.path.join(base_dir, "metrics.json"),
            "label_encoder_path": os.path.join(base_dir, "label_encoder.pkl"),
            "tokenizer_path": os.path.join(base_dir, "tokenizer.pkl"),
            "tensorboard_path": os.path.join(base_dir, "tensorboard"),
            "checkpoint_path": os.path.join(base_dir, "checkpoint"),
        }

    def export_model_plot(self, model):
        tf.keras.utils.plot_model(model, to_file=self.paths['model_plot_path'], show_shapes=True)

    def export_model(self, model):
        model.save_weights(self.paths['model_path'])
        print(f"Model has been exported here => {self.paths['model_path']}")

    def export_encoder(self, label_encoder):
        pickle.dump(label_encoder, open(self.paths['label_encoder_path'], "wb"))
        print(f"Label Encoder has been exported here => {self.paths['label_encoder_path']}")

    def export_info(self, info):
        with open(self.paths['model_info_path'], 'w') as outfile:
            json.dump(info, outfile)
            print(f"Model information have been exported here => {self.paths['model_info_path']}")

    def export_metrics(self, metrics):
        with open(self.paths['metrics_path'], 'w') as outfile:
            json.dump(metrics, outfile)
            print(f"Model metrics have been exported here => {self.paths['metrics_path']}")

    def export_tokenizer(self, tokenizer):
        pickle.dump(tokenizer, open(self.paths['tokenizer_path'], "wb"))
        print(f"Tokenizer has been exported here => {self.paths['tokenizer_path']}")
