from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test):
    """Calculate all six required metrics."""

    predictions = model.predict(X_test)

    # Identify the probability column for class 1 = Malignant
    positive_index = np.flatnonzero(model.classes_ == 1).item()
    probabilities = model.predict_proba(X_test)[:, positive_index]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

    return metrics, predictions


def main():
    # --------------------------------------------------
    # 1. Set paths and create output folders
    # --------------------------------------------------
    project_dir = Path(__file__).resolve().parent

    data_path = project_dir / "training_data.csv"
    model_dir = project_dir / "model"
    results_dir = project_dir / "results"

    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    # 2. Load your CSV
    # --------------------------------------------------
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # Your CSV's first column contains row numbers.
    # Reading it as the index excludes it from the features.
    df = pd.read_csv(data_path, index_col=0)

    if "y" not in df.columns:
        raise ValueError(
            "The dataset must contain the target column 'y'."
        )

    X = df.drop(columns=["y"]).copy()

    # Encode the target:
    # B = Benign -> 0
    # M = Malignant -> 1
    y = df["y"].map({"B": 0, "M": 1})
    y.name = "target"

    # --------------------------------------------------
    # 3. Validate the data
    # --------------------------------------------------
    if y.isna().any():
        raise ValueError(
            "Target column contains missing or unknown labels. "
            "Expected only 'B' and 'M'."
        )

    y = y.astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "The dataset must contain both benign and malignant cases."
        )

    X = X.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Input features contain missing or infinite values."
        )

    if X.shape[1] < 12:
        raise ValueError("At least 12 input features are required.")

    if X.shape[0] < 500:
        raise ValueError("At least 500 instances are required.")

    print("Dataset loaded successfully!")
    print("Feature shape:", X.shape)

    print("\nTarget counts (0 = Benign, 1 = Malignant):")
    print(y.value_counts().sort_index())

    # --------------------------------------------------
    # 4. Split: 80% training and 20% testing
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining features:", X_train.shape)
    print("Testing features:", X_test.shape)

    # --------------------------------------------------
    # 5. Save the held-out test dataset
    # --------------------------------------------------
    test_data = X_test.copy()
    test_data["target"] = y_test

    test_data.to_csv(
        project_dir / "test_data.csv",
        index=False,
    )

    # --------------------------------------------------
    # 6. Define the five classifiers
    # --------------------------------------------------
    # Scaling is fitted only on training data through
    # each pipeline and saved with the classifier.
    models = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]),
        },

        "decision_tree": {
            "name": "Decision Tree",
            "pipeline": Pipeline([
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5,
                        random_state=42,
                    ),
                ),
            ]),
        },

        "knn": {
            "name": "KNN",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(n_neighbors=5),
                ),
            ]),
        },

        "naive_bayes": {
            "name": "Naive Bayes",
            "pipeline": Pipeline([
                ("classifier", GaussianNB()),
            ]),
        },

        "svm": {
            "name": "Support Vector Machine",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]),
        },
    }

    # --------------------------------------------------
    # 7. Train, evaluate and save each model
    # --------------------------------------------------
    comparison_rows = []

    for model_key, model_info in models.items():
        model_name = model_info["name"]
        pipeline = model_info["pipeline"]

        print(f"\nTraining {model_name}...")

        # Learn only from the training portion
        pipeline.fit(X_train, y_train)

        # Evaluate on the same held-out test set
        metrics, predictions = evaluate_model(
            pipeline, X_test, y_test
        )

        comparison_rows.append({
            "Model": model_name,
            **metrics,
        })

        # Save the complete trained pipeline
        model_path = model_dir / f"{model_key}.joblib"
        joblib.dump(pipeline, model_path)

        # Reload and verify saved predictions
        saved_model = joblib.load(model_path)

        if not np.array_equal(
            predictions,
            saved_model.predict(X_test),
        ):
            raise RuntimeError(
                f"Saved-model verification failed for {model_name}."
            )

        # Save confusion matrix
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"],
        )

        matrix_df.to_csv(
            results_dir / f"{model_key}_confusion_matrix.csv"
        )

        # Save classification report
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0,
        )

        report_path = (
            results_dir / f"{model_key}_classification_report.txt"
        )

        report_path.write_text(report, encoding="utf-8")

        print(f"Saved: {model_path.name}")
        print(f"Test accuracy: {metrics['Accuracy']:.4f}")

    # --------------------------------------------------
    # 8. Save the model comparison table
    # --------------------------------------------------
    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        results_dir / "model_comparison.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string(index=False))

    # --------------------------------------------------
    # 9. Save feature schema for the Streamlit app
    # --------------------------------------------------
    schema = {
        "features": X.columns.tolist(),
        "target_column": "target",
        "target_mapping": {
            "0": "Benign",
            "1": "Malignant",
        },
        "random_state": 42,
        "test_size": 0.20,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "models": {
            key: {
                "display_name": info["name"],
                "filename": f"{key}.joblib",
            }
            for key, info in models.items()
        },
    }

    with open(
        model_dir / "feature_schema.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(schema, file, indent=2)

    print("\nTraining completed successfully!")
    print("Saved five trained pipelines in: model/")
    print("Saved evaluation outputs in: results/")
    print("Saved held-out test data as: test_data.csv")


if __name__ == "__main__":
    main()