import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime


def collect_files(location, ext="ipc", raise_empty_exc=False) -> list[str]:
    """
    Get files having from a directory or a single ext file.

    Args:
    location (str): The directory containing IPC files or a single IPC file.

    Returns:
    list: List of file paths having the specified extension.
    """
    if not os.path.exists(location):
        raise FileNotFoundError(f"Location {location} not found")

    if os.path.isdir(location):
        pattern = f"**/*.{ext}"
        file_paths = glob.glob(os.path.join(location, pattern), recursive=True)
        if raise_empty_exc and len(file_paths) == 0:
            raise ValueError(f"No IPC files found in {location}")
    elif os.path.isfile(location) and location.endswith(".ipc"):
        file_paths = [location]
    else:
        raise ValueError(
            f"Location {location} is neither a directory nor an {ext.upper()} file"
        )

    return file_paths


def get_or_create_folder(path: str | Path) -> str:
    assert path is not None, path
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_ipc_files(file_paths):
    dataframes = [pd.read_feather(file_path) for file_path in file_paths]
    return pd.concat(dataframes, ignore_index=True)


def get_timestamp(format="%Y%m%d_%H%M%S"):  # noqa
    """
    Get the current timestamp in the specified format.

    Parameters:
        format (str): Format string for the timestamp (default: "%Y-%m-%d %H:%M:%S").

    Returns:
        str: Formatted timestamp.
    """
    return datetime.now().strftime(format)
