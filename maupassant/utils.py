import time
import functools

import colorful as cf
import numpy as np


def highlight_text_html(text, mark_color='purple'):
    """Create html tag"""
    map_color = {
        "purple": "aa9cfc", "blue": "7aecec", "kaki": "bfe1d9",
        "orange": "feca74", "green": "bfeeb7", "magenta": "aa9cfc"
    }
    padding = "0.45"
    height = "1"
    radius = "0.35"
    cls = "entity"
    style = f'"background:#{map_color[mark_color]};padding:{padding}em;line-height:{height};border-radius:{radius}em">'

    return f' <mark class={cls} style={style}> {text} </mark> '


def timer(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        run_time = time.perf_counter() - start_time
        print(f"{cf.bold_white_on_green}Function:{cf.reset}{cf.bold_magenta}{func.__name__}{cf.reset}")
        print(f"{cf.bold_white_on_green}kwargs:{cf.reset}{cf.bold_magenta}{kwargs}{cf.reset}")
        print(f"{cf.bold_white_on_green}Duration:{cf.reset}{cf.bold_magenta}{run_time*1000:.3f}ms{cf.reset}")

        return value

    return wrapper


def path_logger(path):

    def real_logger(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = func(*args, **kwargs)
            print(f"{cf.bold_white_on_green}{func.__name__}{cf.reset}")
            print(f"{cf.bold_magenta} has been exported: {path}{cf.reset}")

            return value

        return wrapper

    return real_logger


def predict_format(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(kwargs['x'], str):
            kwargs['x'] = np.asarray([kwargs['x']])
        if isinstance(kwargs['x'], list):
            kwargs['x'] = np.asarray(kwargs['x'])

        return func(*args, **kwargs)

    return wrapper


def not_none(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if any(arg is "" for arg in args):
            error = 'ValueError: => function {}: requires at least one word'.format(func.__name__)
            print(f"{cf.bold_red}Error: {error}{cf.reset}")
            return error
        else:
            return func(*args, **kwargs)

    return wrapper
