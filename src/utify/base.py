# -*- coding: utf-8 -*-
import os
import logging
import urllib.request
import warnings
import zipfile
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def strfsec(seconds: int, ndigits=0):
    """Format the time in seconds into a time string

    Args:
        seconds (int): time in seconds
        ndigits (int): number of digits to round the seconds

    Returns:
        str: time string with a nice format
    """
    units = [("day", 24 * 60 * 60), ("hour", 60 * 60), ("minute", 60)]
    res, timestr = seconds, ''
    for unit, value in units:
        if res >= value:
            num = int(res // value)
            timestr += f"{num} {unit}{'s' if num > 1 else ''} "
            res = res % value
    timestr += f"{round(res, ndigits) if ndigits > 0 else int(res)} second{'s' if res > 1 else ''}"
    return timestr


def make_listing(iterable, indent=2, prefix='+', func=str):
    """Convert an iterable to a formatted listing.

    Args:
        iterable: an iterable of items to make the listing
        indent: space size before each item
        prefix: a prefix string to show before each item
        func: use this to customize the appearance of each item

    Returns:

    """
    sep = '\n' + ' ' * indent + prefix + ' '
    return sep + sep.join(func(x) for x in iterable)


def make_divider(text='', char='=', line_max=100, show=False):
    """Print a divider with a headline, e.g.

    ============================ Headline here ===========================

    Args:
        text (unicode): text of headline. If empty, only the chars are printed.
        char (unicode): Line character to repeat, e.g. =.
        line_max (int): max number of chars per line
        show (bool): Whether to print or not.

    Returns:
        str or None: if show, print the divider otherwise return the divider text
    """
    if len(char) != 1:
        raise ValueError(
            "Divider chars need to be one character long. "
            "Received: {}".format(char)
        )

    chars = char * (int(round((line_max - len(text))) / 2) - 2)
    text = " {} ".format(text) if text else ""
    text = f"\n{chars}{text}{chars}"

    if len(text) < line_max:
        text = text + char * (line_max - len(text))
    if show:
        print(text)
    else:
        return text


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, filename=None, overwrite=False, progress=True, **barkws):
    """Download rescource from the given url with an optional progress bar

    Args:
        url (str): url to download
        filename (str): path to save the downloaded file
        overwrite (bool): if True, overwrite the target file if exists
        progress (bool): if True, show progress bar
        **barkws: additional keywords parameters passed to tqdm.tqdm

    Returns:
        None
    """
    filename = filename or url.split('/')[-1]
    if os.path.exists(filename) and (not overwrite):
        msg = f"\ntarget file {filename} exists. Please use overwrite=True to update."
        warnings.warn(msg)
    else:
        if progress:
            barkws = barkws or dict(unit='B', unit_scale=True, miniters=1, desc=f"Download {url.split('/')[-1]}")
            pbar = DownloadProgressBar(**barkws)
            reporthook = pbar.update_to
        else:
            reporthook = None
        urllib.request.urlretrieve(url, filename=filename, reporthook=reporthook)


def unzip(zf, save_dir=None, overwrite=False):
    """Unzip a zip file into a folder

    Args:
        zf (str): filepath of the zip file
        save_dir (str): target folder
        overwrite (bool): if True, overwrite the target folder

    Returns:
        None
    """
    assert os.path.exists(zf), f"{zf} does not exist!"
    save_dir = save_dir or Path(zf).parent.joinpath(Path(zf).name.split('.')[0])
    if os.path.exists(save_dir) and (not overwrite):
        warnings.warn(f"save_dir: {save_dir} exists. Please change save_dir or use overwrite=True.")
    else:
        logger.debug(f"Extracting {zf} into {save_dir}/...")
        with zipfile.ZipFile(zf, 'r') as handle:
            handle.extractall(save_dir)
        logger.debug(f"Successful unzip {zf}")
