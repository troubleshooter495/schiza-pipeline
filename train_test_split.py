from torch.utils.data import random_split
from torch import Generator
import json

from config import SEED


def simple_split(data, train_size: float = 0.8):
    train_share = int(len(data) * train_size)
    train_dataset = data[:train_share]
    test_dataset = data[train_share:]

    return train_dataset, test_dataset


def torch_split(dataset, train_size: float):
    train_share = int(len(dataset) * train_size)
    test_share = len(dataset) - train_share
    train_dataset, test_dataset = random_split(dataset,
                                               [train_share, test_share],
                                               Generator().manual_seed(SEED))
    return train_dataset, test_dataset


def read_from_json(json_path: str):
    with open(json_path) as f:
        kfold_splits = json.load(f)
    return kfold_splits