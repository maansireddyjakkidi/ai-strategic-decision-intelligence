from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN = "y"

# Duration is intentionally excluded because it can introduce
# temporal/information leakage for a pre-contact targeting decision.
EXCLUDED_COLUMNS = {"duration"}

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictor variables from the target.

    The duration variable is deliberately excluded because it represents
    information from the current contact and may not be available at the
    time of an initial targeting decision.

    Parameters
    ----------
    df : pd.DataFrame
        Raw bank marketing dataset.

    Returns
    -------
    X : pd.DataFrame
        Predictor variables.
    y : pd.Series
        Binary target variable.
    """

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
        and column != TARGET_COLUMN
    ]

    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("Target contains unexpected values.")

    return X, y


def introduce_missingness(
    X: pd.DataFrame,
    missing_rate: float,
    random_state: int,
) -> pd.DataFrame:
    """
    Randomly replace an exact proportion of feature cells with NaN.

    Missingness is introduced only into the predictor dataset.
    The target variable is never modified.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor dataset.

    missing_rate : float
        Fraction of predictor cells to replace with missing values.
        Must be between 0 and 1.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of X with controlled missingness.
    """

    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be between 0 and 1.")

    X_degraded = X.copy()

    if missing_rate == 0:
        return X_degraded

    rng = np.random.default_rng(random_state)

    total_cells = X_degraded.shape[0] * X_degraded.shape[1]

    number_to_remove = int(round(total_cells * missing_rate))

    # Create a flat list of every cell position.
    all_positions = np.arange(total_cells)

    # Randomly select unique cell positions.
    selected_positions = rng.choice(
        all_positions,
        size=number_to_remove,
        replace=False,
    )

    # Convert flat positions into row and column positions.
    row_indices, column_indices = np.unravel_index(
        selected_positions,
        X_degraded.shape,
    )

    X_degraded_values = X_degraded.to_numpy(dtype=object)

    X_degraded_values[row_indices, column_indices] = np.nan

    X_degraded = pd.DataFrame(
        X_degraded_values,
        columns=X_degraded.columns,
        index=X_degraded.index,
    )

    return X_degraded

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numerical features:
        Median imputation + standardization.

    Categorical features:
        Most-frequent imputation + one-hot encoding.

    Returns
    -------
    ColumnTransformer
        Scikit-learn preprocessing transformer.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor