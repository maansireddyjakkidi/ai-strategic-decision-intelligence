from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def main():

    summary_path = (
        TABLES_DIR
        / "research_summary.csv"
    )

    statistics_path = (
        TABLES_DIR
        / "statistical_analysis.csv"
    )

    output_path = (
        TABLES_DIR
        / "paper_results_table.csv"
    )

    summary = pd.read_csv(
        summary_path
    )

    statistics = pd.read_csv(
        statistics_path
    )

    # ---------------------------------------------------------------
    # Select baseline results
    # ---------------------------------------------------------------

    baseline = summary[
        summary[
            "requested_missing_rate"
        ] == 0
    ].copy()

    baseline = baseline[
        [
            "model",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
        ]
    ]

    baseline = baseline.rename(
        columns={
            "f1_mean": "baseline_f1",
            "f1_std": "baseline_f1_std",
            "roc_auc_mean": "baseline_roc_auc",
            "roc_auc_std": "baseline_roc_auc_std",
        }
    )

    # ---------------------------------------------------------------
    # Select 20% missingness results
    # ---------------------------------------------------------------

    degraded = summary[
        summary[
            "requested_missing_rate"
        ] == 0.20
    ].copy()

    degraded = degraded[
        [
            "model",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "f1_change_percent",
        ]
    ]

    degraded = degraded.rename(
        columns={
            "f1_mean": "f1_at_20pct",
            "f1_std": "f1_std_at_20pct",
            "roc_auc_mean": "roc_auc_at_20pct",
            "roc_auc_std": "roc_auc_std_at_20pct",
        }
    )

    # ---------------------------------------------------------------
    # Calculate ROC-AUC percentage change
    # ---------------------------------------------------------------

    degraded = degraded.merge(
        baseline[
            [
                "model",
                "baseline_roc_auc",
            ]
        ],
        on="model",
        how="left",
    )

    degraded["roc_auc_change_percent"] = (
        (
            degraded["roc_auc_at_20pct"]
            - degraded["baseline_roc_auc"]
        )
        / degraded["baseline_roc_auc"]
        * 100
    )

    degraded = degraded.drop(
        columns=["baseline_roc_auc"]
    )

    # ---------------------------------------------------------------
    # Add statistical analysis
    # ---------------------------------------------------------------

    paper_table = baseline.merge(
        degraded,
        on="model",
        how="inner",
    )

    paper_table = paper_table.merge(
        statistics,
        on="model",
        how="left",
    )

    # ---------------------------------------------------------------
    # Round values
    # ---------------------------------------------------------------

    numeric_columns = (
        paper_table.select_dtypes(
            include="number"
        ).columns
    )

    paper_table[numeric_columns] = (
        paper_table[numeric_columns]
        .round(4)
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    paper_table.to_csv(
        output_path,
        index=False,
    )

    print("=" * 80)
    print("PAPER RESULTS TABLE")
    print("=" * 80)

    print(
        paper_table.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()