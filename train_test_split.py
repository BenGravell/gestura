import math
import json

import numpy as np

import utils


def write_train_test_idxs(dataset_name="UWaveGestureLibrary", test_size=0.3, random_state=1):
    _, y, _ = utils.load_data(dataset_name)

    n = len(y)
    n_test = math.floor(n * test_size)
    n_train = n - n_test

    random_state_obj = np.random.RandomState(random_state)

    idxs = random_state_obj.permutation(np.arange(n))
    idxs_train = idxs[:n_train]
    idxs_test = idxs[n_train:]

    train_test_idxs = {"idxs_train": idxs_train.tolist(), "idxs_test": idxs_test.tolist()}
    with open("train_test_idxs.json", "w") as file_out:
        json.dump(train_test_idxs, file_out, indent=4)


if __name__ == "__main__":
    write_train_test_idxs()
