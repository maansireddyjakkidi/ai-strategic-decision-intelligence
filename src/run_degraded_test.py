from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_data, validate_data
from models import create_models
from preprocessing import (
    build_preprocessor,
    introduce_missingness,
    prepare_features,
)
from evaluation import evaluate_predictions


RANDOM_STATE = 42
MISSING_RATE = 0.20


def main():
    print("=" * 70)
    print("DEGRADED-DATA TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------

    df = load_data()
    validate_data(df)

    X, y = prepare_features(df)

    # ---------------------------------------------------------------
    # 2. Train/test split
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------------
    # 3. Introduce missingness ONLY into training data
    # ---------------------------------------------------------------

    X_train_degraded = introduce_missingness(
        X_train,
        missing_rate=MISSING_RATE,
        random_state=RANDOM_STATE,
    )

    actual_missingness = (
        X_train_degraded.isna().sum().sum()
        / X_train_degraded.size
    )

    print(
        f"\nRequested missingness: "
        f"{MISSING_RATE:.2%}"
    )

    print(
        f"Actual missingness: "
        f"{actual_missingness:.2%}"
    )

    # ---------------------------------------------------------------
    # 4. Test each model
    # ---------------------------------------------------------------

    models = create_models(
        random_state=RANDOM_STATE
    )

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

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

        print("Training on degraded data...")

        pipeline.fit(
            X_train_degraded,
            y_train,
        )

        print("Evaluating on clean test data...")

        y_pred = pipeline.predict(X_test)

        y_probability = pipeline.predict_proba(
            X_test
        )[:, 1]

        metrics = evaluate_predictions(
            y_test,
            y_pred,
            y_probability,
        )

        for metric_name, value in metrics.items():
            print(
                f"{metric_name:>10}: "
                f"{value:.4f}"
            )

    print("\n" + "=" * 70)
    print("DEGRADED-DATA TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()