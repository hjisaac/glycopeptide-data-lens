#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Script to identify PTMs (Post Translation Modifications) in a AA (Amino Acid) sequences 
i.e., proteins. The PTMs distribution is pretty much like a gaussian distribution.
"""

import os
import re
import sys
import itertools
import logging
from typing import Iterable

# https://stackoverflow.com/questions/17935130/which-module-should-contain-logging-config-dictconfigmy-dictionary-what-about
import logging.config  # noqa
import pandas as pd
from tqdm import tqdm
from enum import Enum
from ordered_set import OrderedSet

# Temporary fix for imports, investigate later
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from common.utils import collect_files, get_timestamp
from common.constants import BASE_RAW_DATA_DIR, BASE_PTMS_DIR, PLUS_INFINITY
from common.logger import get_logger_config


# In[4]:


logger_config = get_logger_config(subdir="scripts")
logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PTMSitesEnum(str, Enum):
    """
    Potential amino acid on which we may have a ptm.
    """

    # Glycosylation only happens on Asparagine within this sequence(Asn-{Any}-Ser/Thr).
    # But we will just assume that it can happen on any Asparagine for simplicity reasons.
    N_GLYCOSYLATION = "N"  # Asparagine (N)
    O_GLYCOSYLATION = "ST"  # Threonine (T) and Serine (S)
    GLYCOSYLATION = N_GLYCOSYLATION + O_GLYCOSYLATION
    ANY = "ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy"  # noqa


GLYCOSYLATION_REGEX_TEMPLATE = r"(?P<aa>[{sites}])\[(?P<glycan_mass>\d+)\]"
N_GLYCOSYLATION_REGEX = GLYCOSYLATION_REGEX_TEMPLATE.format(
    sites=PTMSitesEnum.N_GLYCOSYLATION
)
# Regex to capture any ptm of the nature ABC..[{number}]...
ANY_PTM_REGEX = GLYCOSYLATION_REGEX_TEMPLATE.format(sites=PTMSitesEnum.ANY)

ipc_files = collect_files(BASE_RAW_DATA_DIR)

logger.info(f"Found {len(ipc_files)} IPC files in {BASE_RAW_DATA_DIR}: {ipc_files}")


def identify_ptms(
    ipc_files: list,
    ptm_examples_limit: int = 5,
    keep_duplicated_examples: bool = False,
    return_df=True,  # noqa
) -> dict[str : OrderedSet[tuple]] | pd.DataFrame:
    """
    Identify post-translational modifications (PTMs) from a list of IPC files.

    This function processes a list of IPC files containing peptide sequences
    and extracts PTMs. If a PTM is found, its occurrence is tracked efficiently
    using an OrderedSet just to keep the insertion order.

    Note: After analyzing the data, we can see that the number of ptm is not huge.
    So if the number of examples is not huge, this algo will be efficient in
    terms of memory usage. The data is like few ptms occurs a very huge number
    of times.


    Args:
        ipc_files (list): A list of file paths to IPC files in Feather format.
        ptm_examples_limit (int, optional): The maximum number of examples to store for each PTM.
            Default to 5.

    Returns:
        dict: A dictionary where keys are tuples of glycan mass and site and values are OrderedSets
        containing tuples of (glycan_mass, project_name, file_name, spectrum_id, ipc_index).
    """

    logger.info(f"Starting process with parameters {locals()}")

    # Seen ptms
    seen_ptms = {}

    # As glob lists files folder by folder, keeping track of the
    # previous project helps us to know when we change a project.
    current_project_name = None
    modified_peptides_count = 0
    unmodified_peptides_count = 0
    global_added_examples_count = 0

    for ipc_file in tqdm(ipc_files, desc="Processing IPC files", unit="file"):

        # Grouping the files per project will help to avoid doing this at each iteration
        *_, project_name, file_name = str(ipc_file).split("/")

        if current_project_name != project_name:
            logger.info(f"Start processing the ipc files of the project {project_name}")
            current_project_name = project_name

        df = pd.read_feather(ipc_file)
        # File level added count
        added_examples_count = 0

        for sequence_object in df.itertuples(name="SequenceObject"):
            if sequence_object.modified_peptide is None:
                continue

            # ptms will contain a list of (amino_acid, glycan_mass)
            ptms: list[tuple[str, str]] = re.findall(  # noqa
                # Any ptm with square bracket notation
                N_GLYCOSYLATION_REGEX,  # ANY_PTM_REGEX
                sequence_object.modified_peptide,
            )

            if not ptms:
                # Do nothing if there is no ptm
                logger.debug(
                    f"No ptm found in {sequence_object.modified_peptide}, skipping..."
                )
                unmodified_peptides_count += 1
                continue

            modified_peptides_count += 1

            for ptm in ptms:
                ptm_examples = seen_ptms.get(ptm, OrderedSet())
                if len(ptm_examples) == ptm_examples_limit:
                    # ptm examples count reached so, do nothing
                    continue

                # A hashable datastructures (here tuple) is necessary for OrderSet to work.
                # Index and index of peptide respectively represent df index and spectrum index
                # (amino_acid, glycan_mass, project_name, file_name, spectrum_id, ipc_index)
                ptm_example = (
                    *ptm,
                    project_name,
                    file_name,
                    sequence_object.index,
                    sequence_object.Index,
                    sequence_object.modified_peptide,
                )

                if (not keep_duplicated_examples) and (
                    ptm_example[-1] in {example[-1] for example in ptm_examples}
                ):
                    # This is a known/seen example, so continue
                    logger.debug(f"Skipping already seen ptm example {ptm_example}")
                    continue

                elif len(ptm_examples) == 0:
                    # This is a first-seen example of the ptm
                    seen_ptms[ptm] = OrderedSet([ptm_example])
                    logger.debug(f"Adding first seen example of the ptm {ptm_example}")

                else:
                    seen_ptms[ptm].add(ptm_example)
                    logger.debug(f"Adding example {ptm_example} to seen ptm")

                added_examples_count += 1

        global_added_examples_count += added_examples_count

        logger.info(
            f"Successfully parsed {project_name}/{file_name} ipc file and added "
            f"{added_examples_count} new example from it."
        )

    logger.info(
        f"Process finish with {unmodified_peptides_count} unmodified peptides found, "
        f"{modified_peptides_count} modified peptides found, {len(seen_ptms)} ptms added, "
        f"and {global_added_examples_count} examples added globally."
    )

    return (
        pd.DataFrame(
            itertools.chain.from_iterable(seen_ptms.values()),
            columns=(
                "amino_acid",
                "glycan_mass",
                "project_name",
                "file_name",
                "spectrum_id",
                "ipc_index",
                "modified_peptide",
            ),
        )
        if return_df
        else seen_ptms
    )


# In[ ]:


if __name__ == "__main__":
    csv_name = f"{BASE_PTMS_DIR}/identified_n_glycosylation_ptms_with_all_examples_occurrences{get_timestamp()}.csv"
    ptms_df = identify_ptms(
        ipc_files, ptm_examples_limit=PLUS_INFINITY, keep_duplicated_examples=True
    )
    ptms_df.to_csv(csv_name, index=False)
    logger.info(f"Saved {len(ptms_df)} found ptm examples into {csv_name} successfully")
