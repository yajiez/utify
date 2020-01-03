import sys
import logging
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from textwrap import TextWrapper
from functools import partial
from .base import make_divider


def strwrap(text, width=66, right_padding=True, subsequent_indent=' ', **kwargs):
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
        self.counts = defaultdict(int)

    def info(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.info(msg, *args, **kwargs)
        self.counts['info'] += 1

    def debug(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.debug(msg, *args, **kwargs)
        self.counts['debug'] += 1

    def warning(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.warning(msg, *args, **kwargs)
        self.counts['warning'] += 1

    def error(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.error(msg, *args, **kwargs)
        self.counts['error'] += 1

    def critical(self, msg, *args, **kwargs):
        msg = self.strwrap(msg)
        self.logger.critical(msg, *args, **kwargs)
        self.counts['critical'] += 1

    def good(self, msg):
        msg = self.strwrap(msg)
        self.logger.info(msg, extra={'tag': 'good'})
        self.counts['good'] += 1

    def fail(self, msg):
        msg = self.strwrap(msg)
        self.logger.info(msg, extra={'tag': 'fail'})
        self.counts['fail'] += 1

    def divider(self, msg):
        assert len(msg) < 66, "msg is too long, Please make it shorter than 66."
        msg = make_divider(msg, line_max=70)
        self.logger.info(msg, extra={'tag': 'divider'})
        self.counts['divider'] += 1


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
    def __init__(self):
        super().__init__()

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
        logformat = "%(message)-66s %(asctime)s"

        self.FORMATS = {
            logging.DEBUG: yellow + "🧐 " + logformat + reset,
            logging.INFO: cyan + "📎 " + logformat + reset,
            logging.WARNING: bold_yellow + "🔥 " + logformat + reset,
            logging.ERROR: red + "\u2718  " + logformat + reset,
            logging.CRITICAL: bold_red + "\u2718  " + logformat + reset
        }

        self.EXTRAS = {
            'good': bold_green + "🎉 " + logformat + reset,
            'fail': bold_red + "😞 " + logformat + reset,
            'divider': lightblue + "✨ " + logformat + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        if hasattr(record, 'tag'):
            log_fmt = self.EXTRAS[getattr(record, 'tag')]
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger(name=None, level='INFO', stream=None, logdir=None, logfile=None,
               file_level='DEBUG', mode='a', encoding='utf-8', delay=False, max_width=100):
    """Get a logger for logging

    Args:
        name:
        level:
        stream:
        logdir:
        logfile:
        file_level:
        mode:
        encoding:
        delay:
        max_width:

    Returns:
        logging.Logger: A Python logger for logging outputs
    """
    logger = logging.getLogger(name)
    logger.setLevel(file_level)
    has_stream_handler = False
    has_file_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            has_stream_handler = True
        if isinstance(handler, logging.FileHandler):
            has_file_handler = True

    if not has_file_handler:
        if logdir:
            logdir = Path(logdir)
            logdir.mkdir(exist_ok=True)
            logfile = logfile or f"{name}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.log"
            logfile = logdir.joinpath(logfile)
            if mode == 'w':
                assert not logfile.exists(), f"Logfile {logfile} is already exist! Please use mode='a'"
            fh = logging.FileHandler(logfile, mode=mode, encoding=encoding, delay=delay)
            fh.setLevel(file_level)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(fh)
            datetime_header = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            left = (max_width - len(datetime_header) - 2) // 2
            right = max_width - len(datetime_header) - 2 - left
            logger.info("Start Logging\n" + "=" * left + ' ' + datetime_header + ' ' + "=" * right)

    if not has_stream_handler:
        ch = logging.StreamHandler(stream=stream or sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)

    if not has_file_handler and logfile:
        logger.debug(f"Logs will be saved into {logfile}")

    return TextWrappedLogger(logger)
