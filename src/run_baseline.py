from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_data, validate_data
from models import create_models
from preprocessing import build_preprocessor, prepare_features
from evaluation import evaluate_predictions


RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("BASELINE EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and validate data
    # ------------------------------------------------------------------

    df = load_data()
    validate_data(df)

    print(f"\nDataset shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Prepare features and target
    # ------------------------------------------------------------------

    X, y = prepare_features(df)

    print(f"Predictor shape: {X.shape}")
    print(f"Target size: {y.shape}")

    # ------------------------------------------------------------------
    # 3. Train/test split
    # ------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ------------------------------------------------------------------
    # 4. Create models
    # ------------------------------------------------------------------

    models = create_models(
        random_state=RANDOM_STATE
    )

    # ------------------------------------------------------------------
    # 5. Train and evaluate each model
    # ------------------------------------------------------------------

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

        print("Training...")

        pipeline.fit(
            X_train,
            y_train,
        )

        print("Predicting...")

        y_pred = pipeline.predict(X_test)

        y_probability = pipeline.predict_proba(
            X_test
        )[:, 1]

        metrics = evaluate_predictions(
            y_test,
            y_pred,
            y_probability,
        )

        print("\nResults:")

        for metric_name, value in metrics.items():
            print(
                f"{metric_name:>10}: "
                f"{value:.4f}"
            )


if __name__ == "__main__":
    main()