import sys
import logging
from typing import TextIO, Iterable, Callable
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from textwrap import TextWrapper
from functools import partial
from .base import make_divider
from .base import make_listing


def strwrap(text, width=80, right_padding=False, subsequent_indent=' ', **kwargs):
    """Wrap the text while keeping the original line seperators

    Args:
        text (str): the text to be wrapped
        width (int): max line width
        right_padding (bool): if True, add --- to the right to reach the max width
        subsequent_indent (str): passed to textwrap.TextWrapper
        **kwargs: additional keyword args passed to textwrap.TextWrapper

    Returns:
        str: the wrapped text
    """
    try:
        text = str(text)
    except ValueError as err:
        print(f"{type(text)} can not be converted to a string!")
        raise err
    wrapper = TextWrapper(width=width, subsequent_indent=subsequent_indent, **kwargs)
    lines = text.split('\n')
    wrapped = wrapper.wrap(lines[0])
    if len(lines) > 1:
        for line in lines[1:]:
            wrapped.extend(wrapper.wrap(line))
    if right_padding and (len(wrapped) > 1):
        wrapped[-1] += ' ' + (width - len(wrapped[-1]) - 1) * '-'
        wrapped[-1] += '-' * (len(subsequent_indent) - 2)
    return '\n'.join(wrapped)


class TextWrappedLogger:
    def __init__(self, logger, **wrap_kwargs):
        self.logger = logger
        self.wrap_kwargs = wrap_kwargs or {}
        self.strwrap = partial(strwrap, subsequent_indent='     ', **self.wrap_kwargs)
        self._counts = defaultdict(int)

    def set_stream_level(self, level):
        has_stream_handler = False
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                has_stream_handler = True
                handler.setLevel(level)
                self.logger.info(f"Set the stream handler level to {level}")
        if not has_stream_handler:
            self.warning(f"Logger has no stream handlers")

    def set_file_level(self, level):
        has_file_handler = False
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                has_file_handler = True
                handler.setLevel(level)
                self.logger.info(f"Set the file handler level to {level}")
        if not has_file_handler:
            self.warning(f"Logger has no file handlers")

    def on(self):
        self.logger.disabled = False

    def off(self):
        self.logger.disabled = True

    @property
    def counts(self):
        return dict(self._counts)

    def info(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.info(msg, *args, **kwargs)
        self._counts['info'] += 1

    def debug(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.debug(msg, *args, **kwargs)
        self._counts['debug'] += 1

    def warning(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.warning(msg, *args, **kwargs)
        self._counts['warning'] += 1

    def error(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.error(msg, *args, **kwargs)
        self._counts['error'] += 1

    def critical(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.critical(msg, *args, **kwargs)
        self._counts['critical'] += 1

    def good(self, msg):
        msg = self.strwrap(msg)
        self.logger.info(msg, extra={'tag': 'good'})
        self._counts['good'] += 1

    def fail(self, msg):
        msg = self.strwrap(msg)
        self.logger.info(msg, extra={'tag': 'fail'})
        self._counts['fail'] += 1

    def divider(self, msg=''):
        assert len(msg) < 66, "msg is too long, Please make it shorter than 66."
        msg = make_divider(msg, line_max=88)
        self.logger.info(msg, extra={'tag': 'divider'})
        self._counts['divider'] += 1

    def listing(self, items: Iterable, header: str = '', func: Callable = str, **kwargs):
        header = header or "✨ "
        msg = header + make_listing(items, func=func, **kwargs)
        msg = self.strwrap(msg)
        self.logger.info(msg)
        self._counts['listing'] += 1

    def summary(self):
        if not self.counts:
            self.warning("You have not called this logger. Now you called it once :)")
        else:
            self.divider("Logging Summary")
            self._counts['divider'] -= 1
            self.listing(self.counts.items(), func=lambda i: f"{i[0]:<10s}: {i[1]}")
            self.divider()


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors

    Common used colors:
        "red": 1,
        "green": 2,
        "yellow": 3,
        "blue": 4,
        "pink": 5,
        "cyan": 6,
        "white": 7,
        "grey": 8
    """

    def __init__(self, timestamp=None):
        super().__init__()
        if timestamp:
            assert timestamp in ('left', 'right')
        # grey = "\x1b[38;21m"
        cyan = "\x1b[36;21m"
        # purple = "\x1b[35;21m"
        yellow = "\x1b[33;21m"
        # blue = "\x1b[34;21m"
        lightblue = "\033[94m"
        # green = "\x1b[32;21m"
        bold_yellow = "\x1b[33;1m"
        # bold_blue = "\x1b[34;1m"
        bold_green = "\x1b[32;1m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        # logformat = "%(asctime)s [%(levelname).1s]: %(message)s"
        if timestamp == 'left':
            logformat = "%(asctime)s %(message)-80s"
        elif timestamp == 'right':
            logformat = "%(message)-80s %(asctime)s"
        else:
            logformat = "%(message)-80s"

        self.FORMATS = {
            logging.DEBUG:    yellow + "🧐 " + logformat + reset,
            logging.INFO:     cyan + "📎 " + logformat + reset,
            logging.WARNING:  bold_yellow + "🔥 " + logformat + reset,
            logging.ERROR:    red + " \u2718 " + logformat + reset,
            logging.CRITICAL: bold_red + " \u2718 " + logformat + reset
        }

        self.EXTRAS = {
            'good':    bold_green + "🎉 " + logformat + reset,
            'fail':    bold_red + "😞 " + logformat + reset,
            'divider': lightblue + "✨ " + logformat + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        if hasattr(record, 'tag'):
            log_fmt = self.EXTRAS[getattr(record, 'tag')]
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger(name=None, level='INFO', stream=None, logdir=None, logfile=None,
               file_level='DEBUG', mode='a', encoding='utf-8', delay=False,
               max_width=100, right_padding=False, timestamp=None):
    """Get a logger to start logging

    Args:
        name (str): the name of the logger. If None, return the __main__ logger
        level (str or int): logging level of the logger for the stream handler
        stream (TextIO): if None, use sys.stdout
        logdir (str or Path): if None, now logfile will be created
        logfile (str or Path): if None and logdir is set, then use [name]_[%Y-%m-%dT%H:%M:%S].log
        file_level (str or int): logging level of the logger for the file handler
        mode (str): the mode of the file handler, default 'a'
        encoding (str): encoding of the log file, default UTF-8
        delay (bool): passed to the file handler creation
        max_width (int): the width of the start logging message in the file handler
        right_padding (bool): if True, add padding to the right side of wrapped text
        timestamp (str): can be 'left', 'right', or None

    Returns:
        utify.logger.TextWrappedLogger: A Python logger for logging beautiful messages
    """
    logger = logging.getLogger(name or "__main__")
    logger.setLevel(file_level)

    # Detect and keep existing handlers
    has_stream_handler = False
    has_file_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            has_stream_handler = True
        if isinstance(handler, logging.FileHandler):
            has_file_handler = True

    # Create file handler if not exist
    if (not has_file_handler) and logdir:
        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True)
        if not logfile:
            logfile = f"{name}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.log"
        logfile = logdir.joinpath(logfile)
        if mode == 'w':
            assert not logfile.exists(), f"File {logfile} is already exist! Please use mode='a'"
        fh = logging.FileHandler(logfile, mode=mode, encoding=encoding, delay=delay)
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)

        # Add a datetime header each time the logfile being updated
        datetime_header = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        left = (max_width - len(datetime_header) - 2) // 2
        right = max_width - len(datetime_header) - 2 - left
        logger.info("Start Logging\n" + "=" * left + ' ' + datetime_header + ' ' + "=" * right)
        logger.info(f"Log file: {logfile}")

    if not has_stream_handler:
        ch = logging.StreamHandler(stream=stream or sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(ColoredFormatter(timestamp=timestamp))
        logger.addHandler(ch)

    return TextWrappedLogger(logger, right_padding=right_padding)
