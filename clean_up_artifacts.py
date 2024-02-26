"""Clean up the model and optimizer artifacts to remove obsolete data."""

import os
import argparse
import logging

import paths


def delete_file(file_path):
    try:
        os.remove(file_path)
        logging.info(f"Removed: {file_path}")
    except Exception as exc:
        logging.error(f"Error removing {file_path}: {exc}")


def cleanup(dir_path, live=False):
    filenames = sorted([x.name for x in dir_path.glob("*.pt")])
    filenames_to_delete = filenames[:-1]
    paths_to_delete = [(dir_path / filename).resolve() for filename in filenames_to_delete]
    for path in paths_to_delete:
        if live:
            delete_file(path)
        else:
            # print(f"Cleanup would delete: {path}")
            logging.info(f"Cleanup would delete: {path}")
    logging.info("Process completed!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually delete files. Default is to do a dryrun which only shows which files would be deleted.",
    )
    args = parser.parse_args()
    dir_paths = [paths.MODEL_DIR_PATH, paths.OPTIM_DIR_PATH]
    for dir_path in dir_paths:
        cleanup(dir_path, args.live)
