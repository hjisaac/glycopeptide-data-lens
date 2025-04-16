import os
import glob
import logging
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from .logger import get_logger_config

logger_config = get_logger_config(subdir=None)
logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


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
            f"Location {location} is neither a directory nor a {ext.upper()} file"
        )

    return file_paths


def get_or_create_folder(path: str | Path) -> str:
    assert path is not None, path
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


import pandas as pd
from tqdm import tqdm
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_ipc_files(file_paths, verbose=False, format="ipc") -> tuple[pd.DataFrame, list]:
    dfs = []
    # List to store per-file information
    file_info_list = []

    # Wrap the file_paths iterable with tqdm for a progress bar
    for i, file in tqdm(
        enumerate(file_paths), total=len(file_paths), desc="Processing files"
    ):
        logger.info(f"Processing file {i}: {file}")
        df = pd.read_feather(file) if format in ("ipc", "feather") else pd.read_csv(file)

        # Collect file information
        file_info = {
            "file_path": file,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }

        # Calculate missing values if verbose=True
        total_rows = len(df)
        non_null_counts = df.count()
        missing_values = {}
        for col, non_null_count in non_null_counts.items():
            if non_null_count < total_rows:
                missing_count = total_rows - non_null_count
                missing_values[col] = missing_count
        file_info["missing_values"] = missing_values

        file_info_list.append(file_info)
        dfs.append(df)

    if not dfs:
        logger.info("No valid DataFrames to concatenate.")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    # Log the consolidated file information summary
    if verbose:
        logger.info("=== File Processing Summary ===")
        for i, file_info in enumerate(file_info_list):
            logger.info(f"File {i}: {file_info['file_path']}")
            logger.info(f"  Row count: {file_info['row_count']}")
            logger.info(f"  Column count: {file_info['column_count']}")
            logger.info(f"  Columns: {file_info['columns']}")
            if file_info["missing_values"]:
                logger.info("  Columns with missing values:")
                for col, count in file_info["missing_values"].items():
                    logger.info(
                        f"    Column '{col}': {count} missing values (defined values: {file_info['row_count'] - count}/{file_info['row_count']})"
                    )
            else:
                logger.info("  No columns with missing values.")

    logger.info("Files loading completed.")
    return result, file_info_list


def get_timestamp(format="%Y%m%d_%H%M%S"):  # noqa
    """
    Get the current timestamp in the specified format.

    Parameters:
        format (str): Format string for the timestamp (default: "%Y-%m-%d %H:%M:%S").

    Returns:
        str: Formatted timestamp.
    """
    return datetime.now().strftime(format)
