import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
import json


def read_file_locally(path: str):
    ext = path.split('.')[-1]
    if ext == 'csv':
        file = pd.read_csv(path)
    elif ext == 'par':
        file = pd.read_parquet(path)
    elif ext == 'npy' or ext == 'npz':
        file = np.load(path)
    elif ext == 'mat':
        file = loadmat(path)
    elif ext == 'json':
        with open(path) as f:
            kfold_splits = json.load(f)
        return kfold_splits
    else:
        raise NotImplemented(f'Unknown extension "{ext}"')

    return file


def save_file_locally(file, path: str):
    pass





