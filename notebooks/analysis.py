#!/usr/bin/env python
# coding: utf-8

# # Instanovo-Glyco Training Data Analysis

# In[1]:


import os
import sys
import glob
import logging
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Fix this later, imports should work without this
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


from common.utils import collect_files, get_or_create_folder, load_ipc_files
from common.logger import get_logger_config
from common.constants import (
    BASE_RAW_DATA_DIR,
    BASE_LOGS_DIR,
    BASE_PLOTS_DIR,
    BASE_REPORTS_CSV_DIR,
)


# In[2]:


target_data = "PXD035158"  # "PXD025859"
artifacts_sub_dir = (BASE_RAW_DATA_DIR / target_data).as_posix().split("/")[-1]
logs_dir = get_or_create_folder(BASE_LOGS_DIR / artifacts_sub_dir)
plots_dir = get_or_create_folder(BASE_PLOTS_DIR / artifacts_sub_dir)
csv_dir = get_or_create_folder(BASE_REPORTS_CSV_DIR / artifacts_sub_dir)


# In[3]:


logger_config = get_logger_config(subdir=artifacts_sub_dir)
logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)
logging.warning(
    f"Changing matplotlib font from {plt.rcParams['font.family']} to ['monospace']"
)
plt.rcParams["font.family"] = ["monospace"]


# In[4]:


logger.info(
    f"Starting with artifacts subdir set to {artifacts_sub_dir}, log_dir set to {logs_dir} and plots_dir set to {plots_dir}"
)


#

# In[5]:


# Grab all ipc files of interest but ATTENTION;
# loading all many ipc files will increase the computation time
ipc_files = collect_files(BASE_RAW_DATA_DIR / target_data)
logger.info(f"Found {len(ipc_files)} IPC files")
df, _ = load_ipc_files(ipc_files, verbose=False)
df.head(20)


# In[ ]:


assert False, "Raised intentionally"


# In[6]:


# Graphing functions go here
def save_figure(filename, save_dir=plots_dir):
    """
    Save the current figure to a file.

    Args:
    filename (str): The name of the file to save the figure to.
    save_dir (str): The directory to save the figure in. Default to "../reports/plots/{artifacts_sub_dir}".
    """

    extensions = [".png", ".jpg", ".jpeg", ".pdf"]
    name, extension = os.path.splitext(filename)
    # We are defaulting the extension to "pdf"
    extension = extension.lower() if extension else ".pdf"

    if extension not in extensions:
        raise ValueError(
            f"Unknown extension {extension} from {filename}. Please choose one from {extensions}"
        )

    save_path = os.path.join(save_dir, f"{name}{extension}")
    plt.savefig(save_path, format=extension[1:], bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")


def plot_x_y(
    df,
    index,
    x_column,
    y_column,
    x_label=None,
    y_label=None,
    title=None,
    filename=None,
    save=True,
):
    """
    Plot x and y arrays for a given line in the DataFrame using vertical lines.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    index (int): The index of the line to plot.
    x_column (str): The name of the x column.
    y_column (str): The name of the y column.
    x_label (str): The label for the x-axis. If None, the x_column name is used.
    y_label (str): The label for the y-axis. If None, the y_column name is used.
    title (str): The title of the plot. If None, a default title is used.
    filename (str): The filename to save the plot. If None, the plot is not saved.
    """
    x_values = df.at[index, x_column]
    y_values = df.at[index, y_column]
    plt.figure(figsize=(10, 6))
    plt.vlines(x_values, ymin=0, ymax=y_values, color="b", alpha=0.7)
    plt.scatter(x_values, y_values, color="b")
    x_label = x_label if x_label else x_column
    y_label = y_label if y_label else y_column
    title = title if title else f"{x_label} vs. {y_label} for index {index}"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    if save:
        filename = filename or f"{x_column}_vs_{y_column}_for_index_{index}_xy_plot"
        save_figure(filename)
    plt.show()


def plot_quantitative(
    df, column, xlabel=None, ylabel="Frequency", title=None, filename=None, save=True
):
    """
    Plot histogram and KDE for a quantitative (numeric) column.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to plot.
    label (str): The label to display on the plot. If None, the column name is used.
    """
    xlabel = xlabel if xlabel else column
    title = title if title else f"Histogram and KDE for {xlabel.lower()}"
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    if save:
        filename = filename or (column + "_histogram")
        save_figure(filename=filename)
    plt.show()


def plot_qualitative(
    df,
    column,
    xlabel="Count",
    ylabel=None,
    title=None,
    top_n=20,
    filename=None,
    save=True,
):
    """
    Plot bar plot for a qualitative (categorical) column.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to plot.
    label (str): The label to display on the plot. If None, the column name is used.
    top_n (int): The number of top values to display.
    """
    ylabel = ylabel if ylabel else column
    title = (
        title
        if title
        else (
            f"Top {top_n} most frequent values for {ylabel.lower()}"
            if top_n
            else ylabel.lower()
        )
    )
    top_values = df[column].value_counts().nlargest(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_values.index, x=top_values.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    if save:
        filename = filename or (column + "_barplot")
        save_figure(filename=filename)
    plt.show()


def plot_x_y_bar(
    df,
    column,
    count_column_name,
    xlabel="Duplicate count",
    ylabel="Column combinations",
    title="Duplicate counts for different column combinations",
    xtick_rotation=45,
    filename=None,
    save=True,
):
    """
    Generates a horizontal bar plot for duplicate counts of column combinations.

    Parameters:
    - results_df: pandas DataFrame containing the data.
    - column_name: str, the column name for the combinations to be plotted.
    - count_name: str, the column name for the duplicate counts to be plotted.
    - xlabel: str, label for the x-axis (default is "Duplicate Count").
    - ylabel: str, label for the y-axis (default is "Column Combinations").
    - title: str, the title of the plot (default is "Duplicate Counts for Different Column Combinations").
    - xtick_rotation: int, angle to rotate x-axis ticks (default is 45).
    """
    plt.figure(
        figsize=(12, max(6, len(df) * 0.3))
    )  # Adjust height dynamically based on the number of rows
    plt.barh(
        df[column],
        df[count_column_name],
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=xtick_rotation)
    plt.tight_layout()  # Prevent overlapping
    plt.grid(True, alpha=0.4)
    plt.gca().invert_yaxis()
    if save:
        filename = filename or (column + "_based_duplicates_count_barplot")
        save_figure(filename=filename)
    plt.show()


# ## Columns description
#
# | **Field**               | **Description**                                                                                   |
# |--------------------------|---------------------------------------------------------------------------------------------------|
# | `index`                 | A unique identifier for each entry in the dataset.                                                |
# | `scan`                  | The scan number corresponding to the acquisition of the mass spectrum.                            |
# | `header`                | Additional metadata about the scan, such as file source or retention time.                        |
# | `rt` (Retention Time)   | The time it takes for the peptide to travel through the chromatographic column.                   |
# | `frag_type`             | The type of fragmentation used (e.g., CID, HCD, ETD).                                             |
# | `collision_energy`      | The energy applied to fragment precursor ions.                                                    |
# | `precursor_mz`          | The mass-to-charge ratio (\(m/z\)) of the precursor ion before fragmentation.                     |
# | `precursor_charge`      | The charge state of the precursor ion (e.g., +2, +3).                                             |
# | `precursor_intensity`   | The intensity of the precursor ion signal, reflecting its abundance.                              |
# | `lower_offset`          | The lower bound of the \(m/z\) window for isolating the precursor ion.                            |
# | `upper_offset`          | The upper bound of the \(m/z\) window for isolating the precursor ion.                            |
# | `isolation_target`      | The target \(m/z\) value for isolating the precursor ion.                                         |
# | `mz`                   | The mass-to-charge ratio (\(m/z\)) of ions detected in the spectrum.                              |
# | `intensity`             | The intensity of ions detected in the spectrum, reflecting their abundance.                       |
# | `scale_factor`          | A scaling factor applied to intensities for normalization.                                        |
# | `peptide`               | The sequence of the identified peptide, without modifications.                                    |
# | `modified_peptide`      | The sequence of the peptide with post-translational modifications (e.g., glycosylation).          |
# | `peptide_observed_mz`   | The observed \(m/z\) of the peptide in the spectrum.                                              |
# | `peptide_calc_mz`       | The theoretical \(m/z\) of the peptide based on its sequence and modifications.                   |
# | `delta_mass`            | The difference between observed and calculated mass, indicating potential modifications or errors. |
# | `retention`             | The retention time of the peptide, used for identification confirmation.                          |
# | `expectation`           | A statistical value (e.g., E-value) indicating the reliability of peptide identification.          |
# | `hyperscore`            | A confidence score for peptide identification (e.g., from Mascot or Sequest).                     |
# | `nextscore`             | An additional confidence score for peptide identification.                                        |
# | `probability`           | The probability that the peptide identification is correct.                                       |
# | `auc_intensity`         | The area under the curve (AUC) of the signal intensity, used for quantification.                  |
# | `protein`               | The protein to which the peptide belongs, identified from a database.                             |

# In[6]:


df.info()


# In[7]:


df[["mz"]].iloc[0].mz


# In[8]:


highly_relevant_columns = [
    "peptide",
    "modified_peptide",
    "precursor_mz",
    "precursor_charge",
    "mz",
    "intensity",
    "rt",
    "delta_mass",
    "protein",
]

moderatly_relevant_columns = [
    "scan",
    "header",
    "frag_type",
    "collision_energy",
    "precursor_intensity",
    "peptide_observed_mz",
    "peptide_calc_mz",
    "auc_intensity",
]

less_relevant_columns = [
    "index",
    "lower_offset",
    "upper_offset",
    "isolation_target",
    "scale_factor",
    "expectation",
    "hyperscore",
    "nextscore",
]


# ### Recap statistics for relevant and less relevant columns

# In[9]:


df_described = df[highly_relevant_columns].describe()
df_described.to_csv(csv_dir / "highly_relevant_columns_described_df.csv", index=False)
df_described


# In[10]:


df_described = df[moderatly_relevant_columns].describe()
df_described.to_csv(
    csv_dir / "moderatly_relevant_columns_described_df.csv", index=False
)
df_described


# In[11]:


df_described = df[less_relevant_columns].describe()
df_described.to_csv(csv_dir / "less_relevant_columns_described_df.csv", index=False)
df_described


# In[29]:


value_counts = df[["peptide", "modified_peptide"]].value_counts()
value_counts


# In[34]:


value_counts_df = value_counts.reset_index(name="count")


# In[36]:


# Compute basic statistics for the counts
stats = {
    "mean": value_counts_df["count"].mean(),
    "median": value_counts_df["count"].median(),
    "std_dev": value_counts_df["count"].std(),
    "min": value_counts_df["count"].min(),
    "max": value_counts_df["count"].max(),
    "total": value_counts_df["count"].sum(),
}
stats


# ### Duplicate investigation

# In[ ]:


# Save unique peptides as a single-column CSV
pd.DataFrame({"Unique Peptides": df["peptide"].unique()}).to_csv(
    csv_dir / "unique_peptides.csv", index=False
)

# Save unique modified peptides as a single-column CSV
pd.DataFrame({"Unique Modified Peptides": df["modified_peptide"].unique()}).to_csv(
    csv_dir / "unique_modified_peptides.csv", index=False
)


# In[ ]:


# Investigate duplicates
# assert False, "This code block may take minutes to complete; Do you really want to run this code?, If yes, then disable this assertion."
logger.info(
    "Start replacing list or np.ndarray with tuples for internal comparison purposes"
)


def list_to_tuple_func(value):
    logger.info(f"Transforming {value[:3]} to tuple")
    if isinstance(value, (np.ndarray, list)):
        return tuple(value)
    return value


for column in ("mz", "intensity"):

    # Convert array-like values in the specified columns to tuples. Using
    # another new column will make use of a lot of memory. So, let's just
    # overwrite the values in the specified column.
    logger.info(f"Start list replacement for column {column}")
    df[column] = df[column].apply(list_to_tuple_func)
logger.info("Finish replacing list or np.ndarray with tuples")
# List of columns to consider
columns_to_check = [
    "peptide",
    "modified_peptide",
    "precursor_mz",
    "precursor_charge",
    "mz",
    "intensity",
    # "delta_mass",
    # "protein",
]


# Function to generate aligned labels
def format_label(columns_in_combination):
    formatted = []
    for col in columns_to_check:
        if col in columns_in_combination:
            formatted.append(col.ljust(len(col)))  # Keep column name
        else:
            formatted.append(" " * len(col))  # Add spaces of equal width
    return ", ".join(formatted)  # Use separator for clarity


# Initialize a list to store results
results = []

# Iterate through each combination of column sizes (1-combinaison, 2-combinaison, etc.)
for size in range(1, len(columns_to_check) + 1):
    for comb in itertools.combinations(columns_to_check, size):
        # Count duplicates for the current combination of columns
        duplicate_count = df[list(comb)].duplicated().sum()
        # FIXME: Here .debug should be used
        logger.info(
            f"Combination size: {size}, duplicate count: {duplicate_count}, combinations: {comb}"
        )
        # Store the result as a tuple of (combination, duplicate_count)
        results.append(
            {"columns": format_label(comb), "duplicate_count": duplicate_count}
        )

# Convert results into a DataFrame
results_df = pd.DataFrame(results)

# Print the DataFrame with duplicate counts
print(results_df)

# Plotting the results
plot_x_y_bar(
    results_df,
    "columns",
    "duplicate_count",
    xlabel="Count of duplicates",
    ylabel="Column combinations",
    title="Duplicate counts for different column combinations",
    xtick_rotation=45,
)


# In[ ]:


# Access the mz and intensity of the most abundant peptide and modification
pd.DataFrame({"Unique Peptides": df["peptide"].unique()}).to_csv(
    csv_dir / "unique_peptides.csv", index=False
)

# Save unique modified peptides as a single-column CSV
pd.DataFrame({"Unique Modified Peptides": df["modified_peptide"].unique()}).to_csv(
    csv_dir / "unique_modified_peptides.csv", index=False
)
# most_abundant_rows.head()


# ### We want to see from which proteins do those duplicated peptides come

# In[7]:


top_peptides = df["peptide"].value_counts().nlargest(20).index

# Filter the original DataFrame to only include those peptides
filtered_df = df[df["peptide"].isin(top_peptides)]

# Group by peptide and list all associated proteins
peptide_to_proteins = (
    filtered_df.groupby("peptide")["protein"]
    .apply(lambda proteins: list(set(proteins)))  # optional: remove duplicates
    .reset_index()
)


# In[8]:


filtered_df[["peptide", "protein"]].value_counts()


# In[11]:


# In[13]:


(df["protein"] == "").sum()


# In[19]:


# In[18]:


print(peptide_to_proteins)


# In[ ]:


plot_qualitative(df, "modified_peptide", "Modified peptides")
plot_qualitative(df, "peptide", "Peptide")
plot_qualitative(df, "protein", "Proteins")


# In[ ]:


plot_quantitative(df, "precursor_mz", xlabel="Precursor m/z")
plot_quantitative(df, "precursor_charge", xlabel="Precursor charge")
plot_quantitative(df, "delta_mass", xlabel="Delta mass")


# In[ ]:


peptide_index = 0
plot_x_y(
    df,
    peptide_index,
    "mz",
    "intensity",
    "m/z",
    "Intensity",
    title=f'm/z vs. Intensity for peptide {df.iloc[peptide_index]["peptide"]}',
)


# ## Deep Analysis of modifications

# ### Do we have entries that come with modified_peptide set to peptide without any actual modifications (Indecent cells)?

# In[23]:


projects_dirs = glob.glob(f"{BASE_RAW_DATA_DIR}/*/")
assert projects_dirs, projects_dirs
project_dirs_to_ignore = ["PXD044641_PXD035158"]

empty_modifications_table = []
# TODO: To run, fix the for loop
for project_dir in []:  # projects_dirs:
    project_name = project_dir.split("/")[-2]
    if project_name in project_dirs_to_ignore:
        logger.info(f"Skipping project {project_name} as part of projects to ignore")
        continue
    logger.info(f"Processing project {project_name}")
    project_file_paths = collect_files(location=project_dir, ext="ipc")
    df, _ = load_ipc_files(project_file_paths)
    fake_modification_df = df[df["peptide"] == df["modified_peptide"]]
    empty_modification_count = df["modified_peptide"].isna().sum()
    rows_count = len(df)

    empty_modifications_table.append(
        {
            "Project ID": project_name,
            "Rows count": rows_count,
            "Real modifications count": len(df)
            - empty_modification_count
            - len(fake_modification_df),
            "Empty modifications count": empty_modification_count,
            "Fake modifications count (peptide = modified_peptide)": len(
                fake_modification_df
            ),
        }
    )

modification_df = pd.DataFrame(empty_modifications_table)

modification_df.to_csv(
    BASE_REPORTS_CSV_DIR / "actual_modified_peptide_entries_count_per_project.csv",
    index=False,
)
modification_df.head(8)


# ## Deep missing column/values investigation (Independent section/cells)

# In[ ]:


projects_dirs = glob.glob(f"{BASE_RAW_DATA_DIR}/*/")
assert projects_dirs, projects_dirs
project_dirs_to_ignore = ["PXD044641_PXD035158"]

project_summary = {}


# In[ ]:


# TODO: To run, fix the for loop
for project_dir in []:  # projects_dirs:
    project_name = project_dir.split("/")[-2]
    if project_name in project_dirs_to_ignore:
        logger.info(f"Skipping project {project_name} as part of projects to ignore")
        continue
    project_file_paths = collect_files(location=project_dir, ext="ipc")
    _, summary = load_ipc_files(project_file_paths)
    project_summary[project_name] = summary


# In[9]:


table_rows = []

for project_id, file_info_list in project_summary.items():
    for file_info in file_info_list:
        file_path = file_info["file_path"]
        file_name = file_path.split("/")[-1]  # Extract the file name from the path
        row_count = file_info["row_count"]
        column_count = file_info["column_count"]
        columns = file_info["columns"]
        # isolation_target and modified_peptide are the fields that used to miss
        # That knowledge comes from previous investigations.
        missing_isolation_target = file_info["missing_values"].get(
            "isolation_target", 0
        )
        missing_modified_peptide = file_info["missing_values"].get(
            "modified_peptide", 0
        )

        # Calculate percentages
        percent_missing_isolation_target = (missing_isolation_target / row_count) * 100
        percent_missing_modified_peptide = (missing_modified_peptide / row_count) * 100

        table_rows.append(
            {
                "Project ID": project_id,
                "File Name": file_name,
                "Row Count": row_count,
                "Column Count": column_count,
                "Columns": columns,
                "Missing isolation_target": missing_isolation_target,
                "Missing modified_peptide": missing_modified_peptide,
                "% Missing isolation_target": round(
                    percent_missing_isolation_target, 2
                ),
                "% Missing modified_peptide": round(
                    percent_missing_modified_peptide, 2
                ),
            }
        )


# In[10]:


df_missing_summary = pd.DataFrame(table_rows)

df_missing_summary.to_csv(
    BASE_REPORTS_CSV_DIR / "deep_missing_values_investigation_summary_report.csv",
    index=False,
)


# In[11]:


df_missing_summary.head(10)


# ## PTMs identification

# The scripts `scripts/identify_ptms` helps in identifying ptms.
