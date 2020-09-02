import time
import functools

import numpy as np


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
