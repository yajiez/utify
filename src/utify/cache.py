import json
import os
import warnings
from functools import wraps
from pathlib import Path

import pandas as pd


def cache(file=None, cache_dir='cache', update=False, **save_kwargs):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    else:
        assert cache_dir.is_dir()

    # define the decorator
    def cache_decorator(func):
        if file:
            filename, ext = os.path.splitext(file)
        else:
            filename, ext = func.__name__, ''
        ext = ext or '.pkl'
        filepath = cache_dir.joinpath(f"{filename}{ext}")
        if filepath.exists() and (not update):
            raise ValueError("Cache file exist. Please use update=True to overwrite.")

        @wraps(func)
        def wrapped(*args, **kwargs):
            res = func(*args, **kwargs)
            if (not ext) or (ext == '.pkl'):
                pd.to_pickle(res, filepath, **save_kwargs)
            elif ext == '.json':
                with open(filepath, 'w') as f:
                    json.dump(res, f, **save_kwargs)
            elif ext == '.csv':
                res.to_csv(filepath, **save_kwargs)
            elif ext == '.parquet':
                res.to_parquet(filepath, **save_kwargs)
            else:
                warnings.warn("Unsupported file format. Please save the result manually.")
            return res

        return wrapped

    # return the decorator
    return cache_decorator
