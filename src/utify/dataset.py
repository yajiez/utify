"""Utility functions to load small open datasets
"""
import argparse
import requests
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/yajiez/ml-dataset/master"


class MLDataset(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        s = 'Available attributes:\n  '
        for attr in dir(self):
            if not attr.startswith('_'):
                value = getattr(self, attr)
                if isinstance(value, pd.DataFrame):
                    s += f"{attr}: {value.shape}\n  "
                else:
                    s += f"{attr}: {value}"
        return s


def load_spam():
    data = pd.read_csv(f'{BASE_URL}/spam/spambase.data', header=None)
    colnames = pd.read_csv(f'{BASE_URL}/spam/names.txt', header=None).iloc[:, 0]
    data.columns = list(colnames) + ['label']
    return data


def load_esl_mixture():
    res = requests.get(f"{BASE_URL}/esl_mixture/esl_mixture.json")
    data = res.json()
    data['x'] = pd.DataFrame(data['x'], columns=['x1', 'x2'])
    data['y'] = pd.Series(data['y'])
    return data


def load_forest_covtype():
    res = MLDataset()
    res.data = pd.read_csv(f"{BASE_URL}/covertype/covtype.data.gz", header=None)
    res.attrs = pd.read_csv(f"{BASE_URL}/covertype/attrs.csv")
    res.wildness_areas = pd.read_csv(f"{BASE_URL}/covertype/wildness_areas.csv")
    res.soil_types = pd.read_csv(f"{BASE_URL}/covertype/soil_types.csv")
    res.cover_types = pd.read_csv(f"{BASE_URL}/covertype/cover_types.csv")
    return res

