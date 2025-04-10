#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


from common.utils import collect_files, get_or_create_folder
from common.logger import get_logger_config
from common.constants import (
    BASE_RAW_DATA_DIR,
    BASE_PROCESSED_DATA_DIR,
    BASE_LOGS_DIR,
    BASE_PLOTS_DIR,
    BASE_REPORTS_CSV_DIR,
)


# In[4]:


logger_config = get_logger_config(subdir="scripts")
logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


# In[5]:


# Collect each unique_peptide.csv file
peptides_file_paths = [
    path
    for path in collect_files(BASE_REPORTS_CSV_DIR, ext="csv")
    if "unique_peptides" in path
]

assert peptides_file_paths, peptides_file_paths


# In[7]:


df = pd.concat([pd.read_csv(file) for file in peptides_file_paths], ignore_index=True)
df.head(20)


# In[8]:


df["Unique Peptides"].describe()


# ## Split without Kevin constraint

# In[10]:


unique_peptides_df = df["Unique Peptides"].drop_duplicates()


# In[11]:


indices = np.arange(len(unique_peptides_df))
np.random.shuffle(indices)
split_ratio = 0.8

split_seperator = int(len(unique_peptides_df) * split_ratio)

# Shuffle the DataFrame indices
shuffled_df = unique_peptides_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split
train_peptides_df = shuffled_df.iloc[:split_seperator].reset_index(drop=True)
test_peptides_df = shuffled_df.iloc[split_seperator:].reset_index(drop=True)


# In[12]:


assert len(train_peptides_df) == 35980, len(train_peptides_df)


# In[13]:


assert len(test_peptides_df) == 8996, len(test_peptides_df)


# In[26]:


# In[23]:


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
    project_name: Path | str,
    split_name: Literal["train", "val", "test"],  # noqa
    algorithm_version: Literal["vO", "v1", "v2"],
    potential_peptides_set: set,
    *args,
    max_charge: int = 10,
    **kwargs,
):
    logger.info(f"Instantiating SpectrumDataFrame with args={args} and kwargs={kwargs}")
    sdf = SpectrumDataFrame.load(*args, verbose=True, **kwargs)  # noqa
    logger.info(
        f"Instantiated SpectrumDataFrame with {len(sdf)} spectra from project {project_name}"
    )
    sdf.filter_rows(
        lambda row: (row["precursor_charge"] <= max_charge)
        and (row["precursor_charge"] > 0)
        and (clean_peptide(row["peptide"]) in potential_peptides_set)
    )
    logger.info(f"Got {len(sdf)} spectra after filtering by precursor charge")
    logger.info(f"Starting {split_name} split for project {project_name}")
    target_path = BASE_PROCESSED_DATA_DIR / project_name
    sdf.save(target_path, partition=f"glyco_{algorithm_version}_{split_name}")
    logger.info(
        f"Saved {len(sdf)} spectra for {split_name} to {target_path} for project {project_name}"
    )


# In[16]:


projects_dirs = glob.glob(f"{BASE_RAW_DATA_DIR}/*/")
assert projects_dirs, projects_dirs


# In[17]:


# Version 0 for train/test split


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
            algorithm_version="v0",
            potential_peptides_set=set(peptide_set),
            source=f"{BASE_RAW_DATA_DIR / project_name}/*",
            source_type="ipc",
            column_mapping={"intensity": "intensity_array", "mz": "mz_array"},
        )


# In[18]:


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


# Version 1 for train/test split
logger.info("Starting to split the dataset but taking into account keving suggestion")
dirs_to_ignore = ["PXD044641_PXD035158"]  #
# Focus on massivekb

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
        ("val", set(kevin_val_peptides_array)),
        ("test", set(kevin_test_peptides_array)),
    ]:
        write_split(
            project_name=project_name,
            split_name=split_name,
            algorithm_version="v1",
            potential_peptides_set=set(kevin_peptide_set),
            source=f"{BASE_RAW_DATA_DIR / project_name}/*",
            source_type="ipc",
            column_mapping={"intensity": "intensity_array", "mz": "mz_array"},
        )


# In[9]:


#
# rr = pd.read_parquet(
#     BASE_PROCESSED_DATA_DIR / "PXD035158/dataset-ms-glyco_v1_train-0001-0001.parquet"
# )
# rr.head()


# In[ ]:


# In[ ]:
