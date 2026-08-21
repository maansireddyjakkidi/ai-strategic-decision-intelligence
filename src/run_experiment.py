import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_data, validate_data
from evaluation import evaluate_predictions
from models import create_models
from preprocessing import (
    build_preprocessor,
    introduce_missingness,
    prepare_features,
)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 2026]

MISSING_RATES = [0.00, 0.05, 0.10, 0.20]

TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def run_single_experiment(
    X,
    y,
    model_name,
    model,
    missing_rate,
    seed,
):
    """
    Run one model under one missingness condition and one random seed.
    """

    # ---------------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )

    # ---------------------------------------------------------------
    # Introduce degradation ONLY into training data
    # ---------------------------------------------------------------

    X_train_degraded = introduce_missingness(
        X_train,
        missing_rate=missing_rate,
        random_state=seed,
    )

    actual_missingness = (
        X_train_degraded.isna().sum().sum()
        / X_train_degraded.size
    )

    # ---------------------------------------------------------------
    # Build complete machine-learning pipeline
    # ---------------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    start_time = time.perf_counter()

    pipeline.fit(
        X_train_degraded,
        y_train,
    )

    training_time = (
        time.perf_counter() - start_time
    )

    # ---------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    metrics = evaluate_predictions(
        y_test,
        y_pred,
        y_probability,
    )

    result = {
        "model": model_name,
        "seed": seed,
        "requested_missing_rate": missing_rate,
        "actual_missing_rate": actual_missingness,
        "training_time_seconds": training_time,
        **metrics,
    }

    return result


def main():

    experiment_start = time.perf_counter()

    print("=" * 80)
    print("FULL DATA-QUALITY EXPERIMENT")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print("\nLoading dataset...")

    df = load_data()

    validate_data(df)

    X, y = prepare_features(df)

    print(f"Dataset shape: {df.shape}")
    print(f"Predictor shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    total_runs = (
        len(SEEDS)
        * len(MISSING_RATES)
        * 3
    )

    print(f"\nTotal planned model runs: {total_runs}")

    # ---------------------------------------------------------------
    # Run experiments
    # ---------------------------------------------------------------

    results = []

    run_number = 0

    for seed in SEEDS:

        for missing_rate in MISSING_RATES:

            models = create_models(
                random_state=seed
            )

            for model_name, model in models.items():

                run_number += 1

                print("\n" + "-" * 80)

                print(
                    f"Run {run_number}/{total_runs}"
                )

                print(
                    f"Model: {model_name}"
                )

                print(
                    f"Missingness: "
                    f"{missing_rate:.0%}"
                )

                print(
                    f"Seed: {seed}"
                )

                result = run_single_experiment(
                    X=X,
                    y=y,
                    model_name=model_name,
                    model=model,
                    missing_rate=missing_rate,
                    seed=seed,
                )

                results.append(result)

                print(
                    f"F1: "
                    f"{result['f1']:.4f}"
                )

                print(
                    f"ROC-AUC: "
                    f"{result['roc_auc']:.4f}"
                )

                print(
                    f"Training time: "
                    f"{result['training_time_seconds']:.2f}s"
                )

    # ---------------------------------------------------------------
    # Save individual results
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    raw_results_path = (
        TABLES_DIR
        / "experiment_results.csv"
    )

    results_df.to_csv(
        raw_results_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Calculate summary statistics
    # ---------------------------------------------------------------

    summary_df = (
        results_df
        .groupby(
            [
                "model",
                "requested_missing_rate",
            ]
        )
        [
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "training_time_seconds",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    summary_path = (
        TABLES_DIR
        / "experiment_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Finish
    # ---------------------------------------------------------------

    total_time = (
        time.perf_counter()
        - experiment_start
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal model runs: {len(results_df)}"
    )

    print(
        f"Total execution time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"\nRaw results saved to:"
    )

    print(raw_results_path)

    print(
        f"\nSummary results saved to:"
    )

    print(summary_path)


if __name__ == "__main__":
    main()