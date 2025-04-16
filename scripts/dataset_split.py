#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


def write_split(
    source_dir: Path | str,
    project_name: Path | str,
    split_name: Literal["train", "valid", "test"],  # noqa
    algorithm_version: Literal["v0", "v1", "v2", "v2.1"],
    potential_peptides_set: set,
    max_charge: int = 10,
    drop_unmodified: bool = False,
):
    file_paths = collect_files(location=source_dir)

    sdf, _ = load_ipc_files(file_paths)
    logger.info(f"Loaded {len(sdf)} entries from {source_dir}")

    # Filter by charge and peptide set
    sdf = sdf[
        (sdf["precursor_charge"] <= max_charge)
        & (sdf["precursor_charge"] > 0)
        & (sdf["peptide"].apply(lambda x: clean_peptide(x) in potential_peptides_set))
    ]
    logger.info(f"Got {len(sdf)} spectra after filtering by precursor charge")
    logger.info(f"Starting {split_name} split for project {project_name}")

    # Identify missing and fake modifications
    is_missing = sdf["modified_peptide"].isna()
    is_fake = sdf["modified_peptide"] == sdf["peptide"]

    logger.info(f"Found {is_missing.sum()} rows with missing modified_peptide")
    logger.info(
        f"Found {is_fake.sum()} rows with fake modified_peptide (same as peptide)"
    )

    # Treat fake modifications as unmodified
    is_unmodified = is_missing | is_fake

    if drop_unmodified:
        logger.info("Filtering out rows with missing or fake modified_peptide")
        sdf = sdf[~is_unmodified]
        logger.info(f"Left with {len(sdf)} rows after dropping unmodified rows")
    else:
        logger.info("Filling missing modified_peptide with related peptide")
        sdf.loc[is_missing, "modified_peptide"] = sdf["peptide"]

    assert (
        sdf["precursor_charge"].between(1, max_charge).all()
    ), "Some precursor_charge values are out of range."
    assert all(
        clean_peptide(p) in potential_peptides_set for p in sdf["peptide"]
    ), "Some peptides are not in the allowed set."
    assert (
        sdf["modified_peptide"].isna().sum() == 0
    ), "Every row should have modified_peptide set"

    # Save final file
    target_path = BASE_PROCESSED_DATA_DIR / project_name
    filename = f"dataset-ms-glyco_{algorithm_version}_{split_name}.parquet"
    sdf.to_parquet(path=target_path / filename, index=False)
    logger.info(
        f"Saved {len(sdf)} spectra for {split_name} to {target_path} for project {project_name}"
    )


# In[4]:


projects_dirs = glob.glob(f"{BASE_RAW_DATA_DIR}/*/")
project_dirs_to_ignore = ["PXD044641_PXD035158"]  #

assert projects_dirs, projects_dirs


# In[16]:


logger.info("Starting to split the dataset but using random split")

# Version 0 for train/test split: Constraint free split
# NOTE: This code is broken because of the param drop_unmodified
algorithm_version = "v0"
project_dirs_to_ignore = ["PXD044641_PXD035158"]  #
# DOCME: Replace the [] by projects_dirs to make the to script run
for project_dir in []:  # projects_dirs:
    project_name = project_dir.split("/")[-2]
    if project_name in project_dirs_to_ignore:
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
project_dirs_to_ignore = ["PXD044641_PXD035158"]  #
# Focus on massivekb

algorithm_version = "v1"
# TODO: Uncomment the # projects_dirs to run the script
for project_dir in []:  # projects_dirs:
    project_name = project_dir.split("/")[-2]

    if project_name in project_dirs_to_ignore:
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


# Version 2 or Version 2.1 for train/test/valid split => All rows with missing modified_peptides are filtered out. But is version 2.1 we also filter out fake modifications defined as modifications for which modified_peptide is equal to peptide.
logger.info("Starting to split the dataset but taking into account kevin's suggestion")
project_dirs_to_ignore = ["PXD044641_PXD035158"]  #

# Focus on massivekb
algorithm_version = "v2.1"  # Version 2.1
for project_dir in projects_dirs:  # projects_dirs:
    project_name = project_dir.split("/")[-2]

    if project_name in project_dirs_to_ignore:
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


# ## Attempt to analyze the content of the split files

# ### Split Version 2.1

# In[29]:


split_version = 2.1
logger.info(f"Split version {split_version} content analysis")

for split in ("train", "valid", "test"):
    dfs = []
    for project_dir in projects_dirs:  # projects_dirs:
        project_name = project_dir.split("/")[-2]

        if project_name in project_dirs_to_ignore:
            logger.info(
                f"Skipping project {project_name} as part of projects to ignore"
            )
            continue

        file_path = f"{project_name}/dataset-ms-glyco_v{split_version}_{split}.parquet"
        df = pd.read_parquet(BASE_PROCESSED_DATA_DIR / file_path)
        logger.info(f"Got {len(df)} rows from {file_path} for project {project_name}")
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"Overall {len(result)} rows for project {project_name} {split} v{split_version}"
    )

    result[["peptide", "modified_peptide"]].to_csv(
        BASE_REPORTS_CSV_DIR
        / f"version{split_version}_{split}_split_peptide_and_modified_peptides.csv",
        index=False,
    )


# In[30]:


train_peptide_and_modified_df = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / f"version{split_version}_train_split_peptide_and_modified_peptides.csv",
)

train_peptide_and_modified_df.describe()


# In[31]:


valid_peptide_and_modified_df = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / f"version{split_version}_valid_split_peptide_and_modified_peptides.csv",
)
valid_peptide_and_modified_df.describe()


# In[32]:


test_peptide_and_modified_df = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / f"version{split_version}_test_split_peptide_and_modified_peptides.csv",
)
test_peptide_and_modified_df.describe()


# In[33]:


split_version = 1
logger.info(f"Split version {split_version} content analysis")

for split in ("train", "valid", "test"):
    dfs = []
    for project_dir in projects_dirs:  # projects_dirs:
        project_name = project_dir.split("/")[-2]

        if project_name in project_dirs_to_ignore:
            logger.info(
                f"Skipping project {project_name} as part of projects to ignore"
            )
            continue
        logger.info(f"Reading {project_dir}")
        df = pd.read_parquet(
            "/home/hjisaac/AI4Science/instanovo_instadeep/glycodata_processed_version1/processed"
            f"/{project_name}/dataset-ms-glyco_v{split_version}_{split}.parquet"
        )
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    result[["peptide", "modified_peptide"]].to_csv(
        BASE_REPORTS_CSV_DIR
        / f"version{split_version}_{split}_split_peptide_and_modified_peptides.csv",
        index=False,
    )


# In[34]:


train_peptide_and_modified_df = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / f"version{split_version}_train_split_peptide_and_modified_peptides.csv",
)

train_peptide_and_modified_df.describe()


# In[35]:


valid_peptide_and_modified_df = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / f"version{split_version}_valid_split_peptide_and_modified_peptides.csv",
)
valid_peptide_and_modified_df.describe()


# In[36]:


test_peptide_and_modified_df = pd.read_csv(
    BASE_REPORTS_CSV_DIR
    / f"version{split_version}_test_split_peptide_and_modified_peptides.csv",
)
test_peptide_and_modified_df.describe()


# ## Investigation for Kostas

# In[2]:


train1 = pd.read_csv(
    "/home/hjisaac/AI4Science/instanovo_instadeep/InstanovoGlyco/.trash_local/version1_train_split_peptide_and_modified_peptides.csv"
)


# In[4]:


len(train1["modified_peptide"].unique())


# In[7]:


len(train1)


# In[3]:


train2 = pd.read_csv(
    "/home/hjisaac/AI4Science/instanovo_instadeep/InstanovoGlyco/.trash_local/version2.1_train_split_peptide_and_modified_peptides.csv"
)


#

# In[6]:


len(train2)


# In[5]:


len(train2["modified_peptide"].unique())


#
