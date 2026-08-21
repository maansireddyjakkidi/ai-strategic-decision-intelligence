from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

TABLES_DIR = RESULTS_DIR / "tables"


def main():

    input_path = (
        TABLES_DIR
        / "experiment_results.csv"
    )

    output_path = (
        TABLES_DIR
        / "research_summary.csv"
    )

    df = pd.read_csv(input_path)

    # ---------------------------------------------------------------
    # Mean and standard deviation
    # ---------------------------------------------------------------

    summary = (
        df.groupby(
            [
                "model",
                "requested_missing_rate",
            ]
        )
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            training_time_mean=(
                "training_time_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------------
    # Baseline F1
    # ---------------------------------------------------------------

    baseline = (
        summary[
            summary[
                "requested_missing_rate"
            ]
            == 0
        ][
            [
                "model",
                "f1_mean",
            ]
        ]
        .rename(
            columns={
                "f1_mean": "baseline_f1"
            }
        )
    )

    summary = summary.merge(
        baseline,
        on="model",
        how="left",
    )

    # ---------------------------------------------------------------
    # F1 change
    # ---------------------------------------------------------------

    summary["f1_change"] = (
        summary["f1_mean"]
        - summary["baseline_f1"]
    )

    summary["f1_change_percent"] = (
        summary["f1_change"]
        / summary["baseline_f1"]
        * 100
    )

    # ---------------------------------------------------------------
    # Round numerical values
    # ---------------------------------------------------------------

    numeric_columns = [
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "training_time_mean",
        "baseline_f1",
        "f1_change",
        "f1_change_percent",
    ]

    summary[numeric_columns] = (
        summary[numeric_columns]
        .round(4)
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    summary.to_csv(
        output_path,
        index=False,
    )

    print("=" * 80)
    print("RESEARCH RESULTS SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()