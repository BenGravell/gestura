"""Cleanup the model and optimizer directories to remove obsolete data."""

import os
import argparse

import utils


def delete_files(file_path_list):
    for file in file_path_list:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as exc:
            print(f"Error removing {file}: {exc}")


def cleanup(dir_path, live=False):
    filenames = sorted([x.name for x in dir_path.glob("*.pt")])
    filenames_to_delete = filenames[:-1]
    paths_to_delete = [(dir_path / filename).resolve() for filename in filenames_to_delete]
    if live:
        delete_files(paths_to_delete)
    else:
        for path in paths_to_delete:
            print(f"Cleanup would delete: {path}")
    print("Process completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup the model and optimizer directories to remove obsolete data.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually delete files. Default is to do a dryrun which only shows which files would be deleted.",
    )
    args = parser.parse_args()
    dir_paths = [utils.MODEL_DIR_PATH, utils.OPTIM_DIR_PATH]
    for dir_path in dir_paths:
        cleanup(dir_path, args.live)
