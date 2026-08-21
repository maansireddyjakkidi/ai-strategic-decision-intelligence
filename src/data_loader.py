from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank_marketing"
    / "bank"
    / "bank-full.csv"
)


def load_data() -> pd.DataFrame:
    """
    Load the UCI Bank Marketing dataset.

    Returns
    -------
    pd.DataFrame
        The complete bank marketing dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Perform basic validation of the dataset.
    """

    required_columns = {
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["y"].isna().any():
        raise ValueError("The target variable contains missing values.")


if __name__ == "__main__":
    data = load_data()
    validate_data(data)

    print("Dataset loaded successfully.")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print(f"Target distribution:\n{data['y'].value_counts()}")