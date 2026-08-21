from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_results():
    """Load experiment results."""

    path = (
        TABLES_DIR
        / "experiment_results.csv"
    )

    return pd.read_csv(path)


def create_f1_plot(df):
    """Create F1-score vs missingness plot."""

    plt.figure(figsize=(9, 6))

    for model in df["model"].unique():

        subset = (
            df[df["model"] == model]
            .groupby(
                "requested_missing_rate"
            )["f1"]
            .mean()
        )

        plt.plot(
            subset.index * 100,
            subset.values,
            marker="o",
            label=model,
        )

    plt.xlabel(
        "Artificial Missingness (%)"
    )

    plt.ylabel("Mean F1-score")

    plt.title(
        "Model Performance Under Data Quality Degradation"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "f1_vs_missingness.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def create_roc_auc_plot(df):
    """Create ROC-AUC vs missingness plot."""

    plt.figure(figsize=(9, 6))

    for model in df["model"].unique():

        subset = (
            df[df["model"] == model]
            .groupby(
                "requested_missing_rate"
            )["roc_auc"]
            .mean()
        )

        plt.plot(
            subset.index * 100,
            subset.values,
            marker="o",
            label=model,
        )

    plt.xlabel(
        "Artificial Missingness (%)"
    )

    plt.ylabel("Mean ROC-AUC")

    plt.title(
        "ROC-AUC Under Data Quality Degradation"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "roc_auc_vs_missingness.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def create_training_time_plot(df):
    """Create model training-time comparison."""

    summary = (
        df.groupby("model")[
            "training_time_seconds"
        ]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(9, 6))

    summary.plot(
        kind="bar",
    )

    plt.xlabel("Model")

    plt.ylabel(
        "Mean Training Time (seconds)"
    )

    plt.title(
        "Average Model Training Time"
    )

    plt.xticks(
        rotation=0
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "training_time.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def main():

    print(
        "Loading experiment results..."
    )

    df = load_results()

    print(
        f"Loaded {len(df)} experimental runs."
    )

    create_f1_plot(df)

    create_roc_auc_plot(df)

    create_training_time_plot(df)

    print(
        "\nFIGURE GENERATION COMPLETE"
    )


if __name__ == "__main__":
    main()