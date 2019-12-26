# -*- coding: utf-8 -*-
import logging
import os
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path
import functools
from halo import Halo, HaloNotebook
import numpy as np
import pandas as pd
from IPython.core.display import display
from tqdm import tqdm

from .fastprogress import is_terminal

logger = logging.getLogger(__name__)


def check_file_rows(file_path: str, encoding: str) -> int:
    """Return the number of rows for the given file

    Args:
        file_path (str): absolute or relative file path
        encoding (str): file encoding
    Returns:
        int: number of rows
    """
    return sum(1 for _ in open(file_path, encoding=encoding))


def lazyread(file_path, encoding, skip_rows=None):
    """Return a lazy reader (generator) for the given file

    Args:
        file_path (str): path of the file to read
        encoding (str): file encoding info
        skip_rows (int): number of rows to skip

    Return:
        generator: a generator which yield the file content line by line
    """
    with open(file_path, encoding=encoding) as f:
        if skip_rows:
            for _ in range(skip_rows):
                f.readline()
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    break


def get_h5table_shape(file_path, data_key):
    """Return the shape of a HDF5 table

    Args:
        file_path (str): path of the HDF5 file
        data_key (str): key of the table

    Return:
        tuple: a tuple of numbers of rows and columns
    """
    store = pd.HDFStore(file_path)
    info = store.get_storer(data_key).group.table
    return info.attrs['NROWS'], len(info.colnames)


def get_h5table_cols(file_path, data_key):
    """Return a list of column names for a HDF5 table

    Args:
        file_path (str): path of the HDF5 file
        data_key (str): key of the table

    Return:
        list: a list of all the column names
    """
    store = pd.HDFStore(file_path)
    info = store.get_storer(data_key).group.table
    return list(info.colnames)


def show_h5_info(file_path, show_index=True, head=None):
    """Show useful information about a HDF5 file

    Args:
        file_path (str): path of the HDF5 file
        show_index (bool): whether to show the index information
        head (int): number of head rows to display

    Return:
        None
    """
    store = pd.HDFStore(file_path)
    print('Data Keys:', ', '.join(store.keys()))
    for k in store.keys():
        info = store.get_storer(k).group.table
        make_divider()
        print('Basic information about {}:'.format(k))
        chunk_shape, colnames, nrow = info.chunkshape, info.colnames, info.attrs['NROWS']
        print('Numer of rows: {}'.format(nrow))
        print('{} columns found in this daatset.'.format(len(colnames)))
        print('Chunk shape of storage: ', chunk_shape)
        if show_index:
            col_index_status = info.colindexed
            cols_with_index = []
            cols_without_index = []
            for col, has_index in col_index_status.items():
                if has_index:
                    cols_with_index.append(col)
                else:
                    cols_without_index.append(col)
            print('Indexed cols: ({})'.format(', '.join(cols_with_index)))
            print('Non-indexed cols: ({})'.format(', '.join(cols_without_index)))
        if head is not None:
            print(f'first {head} rows:')
            rows = store.select(k, start=0, stop=head)
            display(rows)
            print('column data types:')
            display(rows.dtypes.to_frame(name='dtype').T)
    store.close()
    make_divider()


def add_h5_index(file_path, force=False):
    """Add index for a HDF5 file

    Args:
        file_path (str): path of the HDF5 file
        force (bool): use force=True to add the index

    Return:
        None
    """
    if force:
        store = pd.HDFStore(file_path, mode='r+')
        for k in store.keys():
            if np.all(list(store.get_storer(k).group.table.colindexed.values())):
                logger.info('\nAll columns in {} have already been indexed. Skipped.'.format(k))
            else:
                logger.info('\nCreating index for all columns in {} ......'.format(k))
                store.create_table_index(k, columns=True)
                logger.info('Task Success!')
        store.close()
    else:
        print("You're modifying the data in-place, make sure you have another copy!")
        print('Use force=True if you really want to do this.')
        print('Task Canceled.')


def get_real_size(obj, visited=None):
    """Recursively caculate the size in bytes of a python object

    Args:
        obj: any python object
        visited (set): only used by recursive calls

    Returns:
        int: object size in bytes
    """
    size = sys.getsizeof(obj)
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return 0
    # important to mark as visited before entering recursion to handle self-referential objects
    visited.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_real_size(v, visited) for v in obj.values()])
        size += sum([get_real_size(k, visited) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_real_size(obj.__dict__, visited)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_real_size(i, visited) for i in obj])
    return size


def approximate_size(size, use_1024_bytes=True):
    """Convert a file size to human-readable form.

    Borrowed from Dive into Python 3 examples.

    Args:
        size (int): file size in bytes
        use_1024_bytes (bool): if True (default), use multiples of 1024, if False, 1000

    Returns:
        str: a string of the file size in a human readable form
    """
    if size < 0:
        raise ValueError('number must be non-negative')

    suffixes = {
        1000: ['KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'],
        1024: ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']
    }

    multiple = 1024 if use_1024_bytes else 1000
    for suffix in suffixes[multiple]:
        size /= multiple
        if size < multiple:
            return '{0:.1f} {1}'.format(size, suffix)

    raise ValueError('number too large')


def memory_usage(obj, deep=False, use_1024_bytes=False):
    """Return memory usage of a pandas DataFrame or Series in human-readable form.

    Args:
        obj (object): any Python object
        deep (bool): args pass into df.memory_usage() method
        use_1024_bytes (bool): if True (default), use multiples of 1024, if False, 1000

    Returns:
        str: a string of the DataFrame size in a human readable form
    """
    if isinstance(obj, pd.DataFrame):
        size = obj.memory_usage(deep=deep).sum()
    elif isinstance(obj, pd.Series):
        size = obj.memory_usage(deep=deep)
    else:
        size = get_real_size(obj) if deep else sys.getsizeof(obj)
    return approximate_size(size, use_1024_bytes=use_1024_bytes)


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


class Spinner(Halo):
    """A better Spinner based on Halo"""

    def __init__(self, text='', clean=False, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.clean = clean

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        if not exc_type:
            if self.clean:
                self.stop()
            else:
                self.succeed(self.text + ' successfully.')
        else:
            self.fail(self.text + ' failed.')

    def __call__(self, func):
        self.text = self.text or ('Running ' + func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class SpinnerNB(HaloNotebook):
    """A better Spinner based on HaloNotebook"""

    def __init__(self, text='', clean=False, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.clean = clean

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        if not exc_type:
            if self.clean:
                self.stop()
            else:
                self.succeed(self.text + ' successfully.')
        else:
            self.fail(self.text + ' failed.')

    def __call__(self, func):
        self.text = self.text or ('Running ' + func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    def stop_and_persist(self, symbol=' ', text=None):
        """Stops the spinner and persists the final frame to be shown.
        Parameters
        ----------
        symbol : str, optional
            Symbol to be shown in final frame
        text: str, optional
            Text to be shown in final frame

        Returns
        -------
        self
        """
        if not self.enabled:
            return self

        output = '\r{} {}\n'.format(*[
            (text, symbol)
            if self._placement == 'right' else
            (symbol, text)
        ][0])
        self.clear()
        print(output)


spinner = Spinner if is_terminal() else SpinnerNB
