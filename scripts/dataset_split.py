#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
from instanovo.utils.data_handler import SpectrumDataFrame

from instanovo.transformer.dataset import remove_modifications as clean_peptide

# Fix this later, imports should work without this
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


# In[18]:


from common.utils import collect_files, get_or_create_folder
from common.logger import get_logger_config
from common.constants import (
    BASE_RAW_DATA_DIR,
    BASE_PROCESSED_DATA_DIR,
    BASE_LOGS_DIR,
    BASE_PLOTS_DIR,
    BASE_REPORTS_CSV_DIR,
)


# In[19]:


logger_config = get_logger_config(subdir="scripts")
logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


# In[20]:


# Collect each unique_peptide.csv file
peptides_file_paths = [
    path
    for path in collect_files(BASE_REPORTS_CSV_DIR, ext="csv")
    if "unique_peptides" in path
]

assert peptides_file_paths, peptides_file_paths


# In[21]:


df = pd.concat([pd.read_csv(file) for file in peptides_file_paths], ignore_index=True)
df.head(20)


# In[22]:


df["Unique Peptides"].describe()


# ## Split without Kevin constraint

# In[23]:


unique_peptides_df = df["Unique Peptides"].drop_duplicates()


# In[24]:


indices = np.arange(len(unique_peptides_df))
np.random.shuffle(indices)
split_ratio = 0.8

split_seperator = int(len(unique_peptides_df) * split_ratio)

# Shuffle the DataFrame indices
shuffled_df = unique_peptides_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split
train_peptides_df = shuffled_df.iloc[:split_seperator].reset_index(drop=True)
test_peptides_df = shuffled_df.iloc[split_seperator:].reset_index(drop=True)


# In[25]:


assert len(train_peptides_df) == 35980, len(train_peptides_df)


# In[26]:


assert len(test_peptides_df) == 8996, len(test_peptides_df)


# In[26]:


# In[27]:


# The zero designs inherited by the SpectrumDataFrame class makes splitting the dataset
# one time complicated.Actually, when we filter an object of the SpectrumDataframe class, the filters
# are kept with the object and are lazily evaluated. So, when an object of the SpectrumDataframe class is filtered, a new object of the that class is not returned, but
# instead it is the old object that is mutated. So if, I decide to use the .filter_rows method, I'll have to filter train and test separately in different inner contexts. But I guess they should be a way to interact with the predicates held by an object of that class.

# v0 => for splitting algorithm without taking into account splitting suggestions from Kevin
# v1 =>


def write_split(
    split_name: "train" | "val" | "test",
    algorithm_version: "vO" | "v1" | "v2",
    potential_peptides_set: set,
    *args,
    max_charge: int = 10,
    **kwargs,
):
    logger.info(f"Instantiating SpectrumDataFrame with args={args} and kwargs={kwargs}")
    sdf = SpectrumDataFrame.load(*args, verbose=True, **kwargs)  # noqa
    logger.info(f"Instantiated SpectrumDataFrame with {len(sdf)} spectra")
    sdf.filter_rows(
        lambda row: (row["precursor_charge"] <= max_charge)
        and (row["precursor_charge"] > 0)
        and (row["peptide"] in potential_peptides_set)
    )
    logger.info(f"Got {len(sdf)} spectra after filtering by precursor charge")
    logger.info(f"Starting train/test splits...")
    target_path = BASE_PROCESSED_DATA_DIR / "PXD035158"
    sdf.save(target_path, partition=f"glyco_{algorithm_version}_{split_name}")
    logger.info(f"Saved {len(sdf)} spectra to {target_path}")
    return sdf


# In[28]:


projects_dirs = glob.glob(f"{BASE_RAW_DATA_DIR}/*/")
assert projects_dirs, projects_dirs


# In[ ]:


# Version 0 for train/test split
for project_dir in projects_dirs:
    project_name = project_dir.split("/")[-2]
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
            split_name=split_name,
            algorithm_version="v0",
            potential_peptides_set=set(train_peptides_df),
            source=f"{BASE_RAW_DATA_DIR / project_name}/*",
            source_type="ipc",
            column_mapping={"intensity": "intensity_array", "mz": "mz_array"},
        )


# In[31]:


# sdf = write_split(
#     split_name="train",
#     algorithm_version="v0",
#     potential_peptides_set=set(train_peptides_df),
#     source=f"{BASE_RAW_DATA_DIR / 'PXD035158'}/*",
#     source_type="ipc",
#     column_mapping={"intensity": "intensity_array", "mz": "mz_array"},
# )


# In[39]:


# In[3]:


#
# rr  = pd.read_parquet(BASE_PROCESSED_DATA_DIR/ "PXD035158/dataset-ms-v0_train-0001-0001.parquet")
# rr.head()


# In[4]:
