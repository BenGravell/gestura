"""Utilities for interacting with JSON data."""

import json
import logging
from pathlib import Path


def to_file(data, file_path: Path | str):
    """
    Writes the given data to a file in JSON format.

    Args:
    - data: The data to write to the file.
    - file_path: The path to the file where the data should be written.

    Returns:
    None
    """

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    logging.debug(f"Data successfully written to {file_path}")


def from_file(file_path: Path | str):
    """
    Reads JSON data from a file and returns it as a dictionary.

    Args:
    - file_path (str): The path to the file from which to read the data.

    Returns:
    dict: The data read from the file.
    """

    with open(file_path) as file:
        data = json.load(file)
        logging.debug(f"Data successfully read from {file_path}")
    return data
