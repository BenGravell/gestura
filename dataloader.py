"""Dataloader."""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from aeon.datasets import load_classification

import constants
import paths
import json_utils
from train_config import TRAIN_CONFIG


class SklearnDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(dataset_name=None):
    if dataset_name is None:
        dataset_name = constants.DATASET_NAME
    return load_classification(dataset_name, extract_path=paths.DATASET_EXTRACT_PATH)


def load_dataset(
    dataset_name=None, swap_seq_dim_axes: bool = True, tensorize: bool = True, return_dataloader: bool = True
):
    X, y, _ = load_data(dataset_name)

    if swap_seq_dim_axes:
        # swap axes so data is organized with dims (num_examples, sequence_length, input_size)
        X = np.swapaxes(X, 1, 2)

    # convert to ints
    y = y.astype(float).astype(int)
    # start at zero for easy indexing
    y = y - min(y)

    if tensorize:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.uint8)

    # Split the data into a training set and a test set
    train_test_idxs = json_utils.from_file(paths.TRAIN_TEST_IDXS_PATH)
    train_idxs = train_test_idxs["idxs_train"]
    test_idxs = train_test_idxs["idxs_test"]

    X_train, y_train = X[train_idxs], y[train_idxs]
    X_test, y_test = X[test_idxs], y[test_idxs]

    datasets = (SklearnDataset(X_train, y_train), SklearnDataset(X_test, y_test))
    if return_dataloader:
        return (DataLoader(dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=True) for dataset in datasets)
    return datasets
