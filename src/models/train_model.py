import os
import pathlib
import pickle
import sys
import yaml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Go up two levels: src/models -> src -> insurance (root)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now this import will find 'src' without error
from src.features.build_features import InsuranceFeatureEngineer
# (Note: adjust the path above to match where your class lives, e.g., src.features or src.features.build_features)

def build_pipeline(n_estimators=100, max_depth=None, random_state=42):
    categorical_features = ["age_group", "lifestyle_risk", "occupation", "city_tier"]
    numeric_features = ["bmi", "income_lpa"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", InsuranceFeatureEngineer()),
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline


def main():
    home_dir = ROOT_DIR

    # Load configuration
    params_file = os.path.join(home_dir, "params.yaml")
    with open(params_file, "r") as f:
        params = yaml.safe_load(f).get("train_model", {})

    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", None)
    seed = params.get("seed", 42)

    # Paths
    processed_dir = os.path.join(home_dir, "data", "processed")
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    models_dir = os.path.join(home_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_output_path = os.path.join(models_dir, "model.pkl")

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "insurance_premium_category"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Train
    pipeline = build_pipeline(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))

    # Save artifact
    with open(model_output_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved successfully to: {model_output_path}")


if __name__ == "__main__":
    main()