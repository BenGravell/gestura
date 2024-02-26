"""Paths."""

import os
import logging
from pathlib import Path
import tempfile


ARTIFACTS_PATH = Path("artifacts")
MODEL_DIR_PATH = ARTIFACTS_PATH / "model"
OPTIM_DIR_PATH = ARTIFACTS_PATH / "optimizer"

DATASET_CONFIG_PATH = Path("dataset_config")
TRAIN_TEST_IDXS_PATH = DATASET_CONFIG_PATH / "train_test_idxs.json"


def make_tmp_dir():
    tmp_dir = tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    logging.debug(f"Created tmp dir at {tmp_dir}")
    return tmp_dir


ROOT_TMP_DIR = make_tmp_dir()

DATASET_EXTRACT_PATH = os.path.join(ROOT_TMP_DIR, "aeon", "datasets")
