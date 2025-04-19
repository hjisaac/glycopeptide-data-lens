from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
BASE_RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
BASE_PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
BASE_REPORTS_DIR = ROOT_DIR / "reports"
BASE_REPORTS_CSV_DIR = ROOT_DIR / "reports" / "csv_misc"
BASE_LOGS_DIR = ROOT_DIR / "reports" / "logs"
BASE_PLOTS_DIR = ROOT_DIR / "reports" / "plots"
BASE_PTMS_DIR = ROOT_DIR / "reports" / "ptms"

IDENTITY_FILE_PATHS = [
    BASE_REPORTS_CSV_DIR / "identity_splits_proteome_tools_from_kevin.csv",
    BASE_REPORTS_CSV_DIR / "identity_splits_massivekb_from_kevin.csv",
    BASE_REPORTS_CSV_DIR / "identity_splits_phospho_from_kevin.csv",
    BASE_REPORTS_CSV_DIR / "identity_splits_pride.csv",
]
BLACKLIST_FILE_PATHS = [
    BASE_REPORTS_CSV_DIR / "identity_splits_blacklist_from_kevin.csv",
]
