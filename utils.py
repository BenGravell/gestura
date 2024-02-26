"""Utilities."""

from typing import Any
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np


def get_timestamp_millis() -> int:
    return int(datetime.timestamp(datetime.now()) * 1000)


def get_hash(data: Any) -> str:
    return hashlib.md5(str(data).encode()).hexdigest()


def get_most_recent_path(dir_path: Path, glob_pattern: str):
    filenames = [x.name for x in dir_path.glob(glob_pattern)]
    timestamps = [int(filename.split("_")[1]) for filename in filenames]
    latest_timestamp_idx = np.argmax(timestamps)
    latest_timestamp_filename = filenames[latest_timestamp_idx]
    return dir_path / latest_timestamp_filename
