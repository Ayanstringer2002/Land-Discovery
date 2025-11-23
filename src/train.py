import argparse
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from preprocess import load_and_concat, build_composite_and_label


def main(data_paths):

    # 1. Load and combine data
    print("\nLoading datasets...")
    df = load_and_concat(data_paths)
    print("Total records:", len(df))

    # 2. Preprocess + generate composite score + labels
    print("\nGenerating features and labels...")
    df, used_columns = build_composite_and_label(df)

    safety_col, infra_col, env_col = used_columns
    print("Columns used:")
    print("Safety:", safety_col)
    print("Infrastructure:", infra_col)
    print("Environment:", env_col)

    # 3. Save processed dataset
    os.makedirs("data", exist_ok=True)
    processed_path = "data/combined_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved at: {processed_path}")

    # 4. Prepare training data
    features = ["_safety_n", "_infra_n", "_env_n", "_composite_score"]
    target = "recommendation_label"

    X = df[features].fillna(0)
    y = df[target]

    # 5. Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")

    # 6. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 7. Initialize models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    best_score = 0
    best_model_name = None
    best_model = None

    print("\nTraining models...")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # Save each model
        model_path = f"models/{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model at: {model_path}")

        # Track best model
        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_model = model

    # Save best model separately
    joblib.dump(best_model, "models/best_model.joblib")

    print("\n✅ Training Complete!")
    print("Best Model:", best_model_name)
    print("Best Accuracy:", best_score)
    print("Saved as: models/best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Land Recommendation ML Models")

    parser.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        help="Paths to all city CSV files (Delhi, Mumbai, etc.)"
    )

    args = parser.parse_args()

    main(args.data_paths)
