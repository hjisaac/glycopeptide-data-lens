#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal
from pathlib import Path
from instanovo.utils.data_handler import SpectrumDataFrame

from instanovo.transformer.dataset import remove_modifications as clean_peptide

# Fix this later, imports should work without this
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


# In[8]:


from common.utils import collect_files, get_or_create_folder, load_ipc_files
from common.logger import get_logger_config
from common.constants import (
    BASE_RAW_DATA_DIR,
    BASE_PROCESSED_DATA_DIR,
    BASE_LOGS_DIR,
    BASE_PLOTS_DIR,
    ROOT_DIR,
    BASE_REPORTS_CSV_DIR,
)


# In[3]:


logger_config = get_logger_config(subdir="scripts")
logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


# In[4]:


# Collect each unique_peptide.csv file
peptides_file_paths = [
    path
    for path in collect_files(BASE_REPORTS_CSV_DIR, ext="csv")
    if "unique_peptides" in path
]

assert peptides_file_paths, peptides_file_paths


# In[14]:


df = pd.concat([pd.read_csv(file) for file in peptides_file_paths], ignore_index=True)
df.head(20)


# In[15]:


df["Unique Peptides"].describe()


# ## Split without Kevin constraint

# In[16]:


unique_peptides_df = df["Unique Peptides"].drop_duplicates()


# In[18]:


indices = np.arange(len(unique_peptides_df))
np.random.shuffle(indices)
split_ratio = 0.8

split_seperator = int(len(unique_peptides_df) * split_ratio)

# Shuffle the DataFrame indices
shuffled_df = unique_peptides_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split
train_peptides_df = shuffled_df.iloc[:split_seperator].reset_index(drop=True)
test_peptides_df = shuffled_df.iloc[split_seperator:].reset_index(drop=True)


# In[19]:


assert len(train_peptides_df) == 35980, len(train_peptides_df)
assert len(test_peptides_df) == 8996, len(test_peptides_df)


# In[20]:


# The zero-copy designs inherited by the SpectrumDataFrame class makes splitting the dataset
# one time complicated. Actually, when we filter an object of the SpectrumDataframe class, the filters
# are kept with the object and are lazily evaluated. So, when an object of the SpectrumDataframe class
# is filtered, a new object of the that class is not returned, but instead it is the old object that
# is mutated. So if, I decide to use the .filter_rows method, I'll have to filter train and test separately
# in different inner contexts. But I guess they should be a way to interact with the predicates held by
# an object of that class.

# v0 => for splitting algorithm without taking into account splitting suggestions from Kevin
# v1 =>


def write_split(
    source_dir: Path | str,
    project_name: Path | str,
    split_name: Literal["train", "valid", "test"],  # noqa
    algorithm_version: Literal["v0", "v1", "v2"],
    potential_peptides_set: set,
    max_charge: int = 10,
    drop_unmodified: bool = False,
):

    file_paths = collect_files(location=source_dir)

    sdf = load_ipc_files(file_paths)
    logger.info(f"Loaded {len(sdf)} entries from {source_dir}")
    sdf = sdf[
        (sdf["precursor_charge"] <= max_charge)
        & (sdf["precursor_charge"] > 0)
        & (sdf["peptide"].apply(lambda x: clean_peptide(x) in potential_peptides_set))
    ]
    logger.info(f"Got {len(sdf)} spectra after filtering by precursor charge")
    logger.info(f"Starting {split_name} split for project {project_name}")

    missing_count = sdf["modified_peptide"].isna().sum()
    logger.info(f"Found {missing_count} rows with missing modified_peptide")

    if drop_unmodified:
        logger.info("Filtering out rows with missing modified_peptide")
        # Filter out rows with missing modified_peptide
        sdf = sdf.dropna(subset=["modified_peptide"])
        logger.info(
            f"Left with {len(sdf)} rows after dropping rows with missing modified_peptide"
        )
    else:
        logger.info("Filling in missing modified_peptide with related peptide")
        # Use peptide as default value for modified_peptide
        sdf["modified_peptide"] = sdf["modified_peptide"].fillna(sdf["peptide"])
    assert (
        sdf["precursor_charge"].between(1, max_charge).all()
    ), "Some precursor_charge values are out of range."
    assert all(
        clean_peptide(p) in potential_peptides_set for p in sdf["peptide"]
    ), "Some peptides are not in the allowed set."
    assert (
        sdf["modified_peptide"].isna().sum() == 0
    ), "Every row should have modified_peptide set"

    target_path = BASE_PROCESSED_DATA_DIR / project_name
    filename = f"dataset-ms-glyco_{algorithm_version}_{split_name}.parquet"
    sdf.to_parquet(path=target_path / filename, index=False)
    logger.info(
        f"Saved {len(sdf)} spectra for {split_name} to {target_path} for project {project_name}"
    )


# In[21]:


projects_dirs = glob.glob(f"{BASE_RAW_DATA_DIR}/*/")
assert projects_dirs, projects_dirs


# In[22]:


# Version 0 for train/test split: Constraint free split
# NOTE: This code is broken because of the param drop_unmodified
algorithm_version = "v0"
dirs_to_ignore = ["PXD044641_PXD035158"]  #
# DOCME: Replace the [] by projects_dirs to make the to script run
for project_dir in []:  # projects_dirs:
    project_name = project_dir.split("/")[-2]
    if project_name in dirs_to_ignore:
        logger.info(f"Skipping project {project_name} as part of projects to ignore")
        continue
    project_file_paths = collect_files(location=project_dir, ext="ipc")

    logger.info(
        f"Collected {len(project_file_paths)} of project {project_name} files from {project_dir}"
    )
    for split_name, peptide_set in [
        ("train", set(train_peptides_df)),
        # ("val", set(val_peptides_df)),
        ("test", set(test_peptides_df)),
    ]:
        write_split(
            project_name=project_name,
            split_name=split_name,
            algorithm_version=algorithm_version,
            potential_peptides_set=set(peptide_set),
            source_dir=f"{BASE_RAW_DATA_DIR / project_name}/",
        )


# In[23]:


kevin_train_peptides_array = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / "train_blacklist_overlap_identity_splits_massivekb_from_kevin_1067866_with_glyco_projects_44976_found_15499.csv"
)["Overlapped train peptides"].unique()
kevin_test_peptides_array = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / "test_overlap_identity_splits_massivekb_from_kevin_33575_with_glyco_projects_44976_found_4136.csv"
)["Overlapped test peptides"].unique()
kevin_val_peptides_array = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / "valid_overlap_identity_splits_massivekb_from_kevin_13062_with_glyco_projects_44976_found_495.csv"
)["Overlapped valid peptides"].unique()


# In[ ]:


# Version 1 for train/test/valid split => peptide is used as fallback for modified_peptide
logger.info("Starting to split the dataset but taking into account kevin's suggestion")
dirs_to_ignore = ["PXD044641_PXD035158"]  #
# Focus on massivekb

algorithm_version = "v1"
# TODO: Uncomment the # projects_dirs to run the script
for project_dir in []:  # projects_dirs:
    project_name = project_dir.split("/")[-2]

    if project_name in dirs_to_ignore:
        logger.info(f"Skipping project {project_name} as part of projects to ignore")
        continue
    project_file_paths = collect_files(location=project_dir, ext="ipc")

    logger.info(
        f"Collected {len(project_file_paths)} of project {project_name} files from {project_dir}"
    )

    for split_name, kevin_peptide_set in [
        ("train", set(kevin_train_peptides_array)),
        ("valid", set(kevin_val_peptides_array)),
        ("test", set(kevin_test_peptides_array)),
    ]:
        write_split(
            drop_unmodified=False,
            project_name=project_name,
            split_name=split_name,
            algorithm_version=algorithm_version,
            potential_peptides_set=set(kevin_peptide_set),
            source_dir=f"{BASE_RAW_DATA_DIR / project_name}/",
        )


# In[5]:


# Version 2 for train/test/valid split => All rows with missing modified_peptides are filtered out.
logger.info("Starting to split the dataset but taking into account kevin's suggestion")
dirs_to_ignore = ["PXD044641_PXD035158"]  #
# Focus on massivekb

algorithm_version = "v2"
for project_dir in projects_dirs:  # projects_dirs:
    project_name = project_dir.split("/")[-2]

    if project_name in dirs_to_ignore:
        logger.info(f"Skipping project {project_name} as part of projects to ignore")
        continue
    project_file_paths = collect_files(location=project_dir, ext="ipc")

    logger.info(
        f"Collected {len(project_file_paths)} of project {project_name} files from {project_dir}"
    )

    for split_name, kevin_peptide_set in [
        ("train", set(kevin_train_peptides_array)),
        ("valid", set(kevin_val_peptides_array)),
        ("test", set(kevin_test_peptides_array)),
    ]:
        write_split(
            # The difference here
            drop_unmodified=True,
            project_name=project_name,
            split_name=split_name,
            algorithm_version=algorithm_version,
            potential_peptides_set=set(kevin_peptide_set),
            source_dir=f"{BASE_RAW_DATA_DIR / project_name}/",
        )


# In[9]:


# rr["modified_peptide"].head(100)


# In[ ]:


# Attempt to analyze the content of the files


# In[5]:


result = pd.read_parquet(
    BASE_PROCESSED_DATA_DIR
    / f"PXD025859/dataset-ms-glyco_{algorithm_version}_train.parquet"
)
result[["peptide", "modified_peptide"]].to_csv(
    ROOT_DIR
    / f".trash_local/version{algorithm_version.replace('v', '')}_split_peptide_and_modified_peptides.csv",
    index=False,
)


#
