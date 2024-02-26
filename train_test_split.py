"""Script to split a dataset into training and testing sets."""

import logging
import math

import numpy as np

import dataloader
import json_utils
import dataset_config
import paths


def create_train_test_idxs():
    y = dataloader.load_data()[1]

    n = len(y)
    n_test = math.floor(n * dataset_config.TEST_FRACTION)
    n_train = n - n_test

    random_state_obj = np.random.RandomState(dataset_config.RANDOM_STATE)

    idxs = random_state_obj.permutation(np.arange(n))
    idxs_train = idxs[:n_train]
    idxs_test = idxs[n_train:]

    train_test_idxs = {"idxs_train": idxs_train.tolist(), "idxs_test": idxs_test.tolist()}
    json_utils.to_file(train_test_idxs, paths.TRAIN_TEST_IDXS_PATH)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    create_train_test_idxs()
