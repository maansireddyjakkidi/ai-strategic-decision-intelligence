from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


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
        / "statistical_analysis.csv"
    )

    df = pd.read_csv(input_path)

    results = []

    # ---------------------------------------------------------------
    # Analyze each model separately
    # ---------------------------------------------------------------

    for model in df["model"].unique():

        model_df = df[
            df["model"] == model
        ].copy()

        # Relationship between missingness and F1
        f1_correlation, f1_pvalue = spearmanr(
            model_df["requested_missing_rate"],
            model_df["f1"],
        )

        # Relationship between missingness and ROC-AUC
        auc_correlation, auc_pvalue = spearmanr(
            model_df["requested_missing_rate"],
            model_df["roc_auc"],
        )

        results.append(
            {
                "model": model,
                "f1_spearman_rho": f1_correlation,
                "f1_p_value": f1_pvalue,
                "roc_auc_spearman_rho": auc_correlation,
                "roc_auc_p_value": auc_pvalue,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.round(4)

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()