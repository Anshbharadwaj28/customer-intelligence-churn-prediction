
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = (PROJECT_ROOT/ "data"/ "raw"/ "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(data_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the original customer churn dataset.

    Args:
        data_path: Path to the raw CSV dataset.

    Returns:
        pd.DataFrame: Original customer dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset is empty.
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df
