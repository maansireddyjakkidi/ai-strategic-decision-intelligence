from data_loader import load_data, validate_data
from preprocessing import (
    build_preprocessor,
    introduce_missingness,
    prepare_features,
)


def main():
    print("Loading dataset...")

    df = load_data()
    validate_data(df)

    print(f"Original shape: {df.shape}")

    X, y = prepare_features(df)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target mean: {y.mean():.4f}")

    print("\nIntroducing 10% missingness...")

    X_degraded = introduce_missingness(
        X,
        missing_rate=0.10,
        random_state=42,
    )

    missing_percentage = (
        X_degraded.isna().sum().sum()
        / X_degraded.size
        * 100
    )

    print(
        f"Actual missing percentage: "
        f"{missing_percentage:.2f}%"
    )

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_degraded)

    print(
        f"Transformed data shape: "
        f"{transformed.shape}"
    )

    print("\nPREPROCESSING TEST PASSED")


if __name__ == "__main__":
    main()