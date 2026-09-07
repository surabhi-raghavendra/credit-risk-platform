"""Load and combine the Home Credit source datasets."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger


logger = get_logger(__name__)
DATA_DIR = PROJECT_ROOT / "data"


def _read_csv(path: str | Path | None, default_name: str) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DATA_DIR / default_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info("Loading %s", csv_path)
    return pd.read_csv(csv_path)


def _require_columns(
    df: pd.DataFrame, columns: Iterable[str], dataset_name: str
) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def load_application_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the main application-level training data."""
    applications = _read_csv(path, "application_train.csv")
    _require_columns(applications, ["SK_ID_CURR"], "application data")
    return applications


def load_bureau_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load and aggregate bureau records to one row per applicant."""
    bureau = _read_csv(path, "bureau.csv")
    required = ["SK_ID_CURR", "AMT_CREDIT_SUM", "CREDIT_DAY_OVERDUE"]
    _require_columns(bureau, required, "bureau data")

    bureau = bureau.copy()
    bureau["_is_overdue"] = bureau["CREDIT_DAY_OVERDUE"].fillna(0).gt(0).astype(int)

    return (
        bureau.groupby("SK_ID_CURR", as_index=False)
        .agg(
            bureau_loan_count=("SK_ID_CURR", "size"),
            bureau_avg_credit_sum=("AMT_CREDIT_SUM", "mean"),
            bureau_overdue_count=("_is_overdue", "sum"),
        )
    )


def load_previous_application_data(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and aggregate previous applications to one row per applicant."""
    previous = _read_csv(path, "previous_application.csv")
    required = ["SK_ID_CURR", "AMT_APPLICATION", "NAME_CONTRACT_STATUS"]
    _require_columns(previous, required, "previous application data")

    previous = previous.copy()
    previous["_is_approved"] = (
        previous["NAME_CONTRACT_STATUS"].eq("Approved").astype(int)
    )

    return (
        previous.groupby("SK_ID_CURR", as_index=False)
        .agg(
            previous_application_count=("SK_ID_CURR", "size"),
            previous_avg_amount=("AMT_APPLICATION", "mean"),
            previous_approval_rate=("_is_approved", "mean"),
        )
    )


def build_master_dataset(
    application_path: str | Path | None = None,
    bureau_path: str | Path | None = None,
    previous_application_path: str | Path | None = None,
) -> pd.DataFrame:
    """Left-join bureau and previous-application aggregates to applications."""
    applications = load_application_data(application_path)
    bureau = load_bureau_data(bureau_path)
    previous = load_previous_application_data(previous_application_path)

    master = applications.merge(
        bureau, on="SK_ID_CURR", how="left", validate="one_to_one"
    )
    master = master.merge(
        previous, on="SK_ID_CURR", how="left", validate="one_to_one"
    )

    count_columns = [
        "bureau_loan_count",
        "bureau_overdue_count",
        "previous_application_count",
    ]
    master[count_columns] = master[count_columns].fillna(0).astype("int64")

    logger.info("Built master dataset with %d rows and %d columns", *master.shape)
    return master
