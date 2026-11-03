from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# --------------------------------------------------
# 1. Page settings
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Model Explorer",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"


# --------------------------------------------------
# 2. Functions to load saved files
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def calculate_metrics(model, X, y):
    predictions = model.predict(X)

    positive_index = np.flatnonzero(model.classes_ == 1).item()
    probabilities = model.predict_proba(X)[:, positive_index]

    auc = (
        roc_auc_score(y, probabilities)
        if y.nunique() == 2
        else np.nan
    )

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(
            y, predictions, pos_label=1, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, pos_label=1, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, pos_label=1, zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions),
    }

    return metrics, predictions, probabilities


# --------------------------------------------------
# 3. Introduction
# --------------------------------------------------
st.title("Breast Cancer Model Explorer")

st.write(
    "Upload a labelled test dataset to compare five trained "
    "classification models."
)

st.caption(
    "Target labels: 0 = Benign, 1 = Malignant. "
    "This app demonstrates an academic classification experiment."
)


# --------------------------------------------------
# 4. Read the saved feature schema
# --------------------------------------------------
if not SCHEMA_PATH.exists():
    st.error(
        "Feature schema is missing. Run train_models.py first."
    )
    st.stop()

try:
    with open(SCHEMA_PATH, encoding="utf-8") as file:
        schema = json.load(file)
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Could not read the feature schema: {error}")
    st.stop()

feature_names = schema["features"]
target_column = schema["target_column"]
model_details = schema["models"]

missing_models = [
    info["filename"]
    for info in model_details.values()
    if not (MODEL_DIR / info["filename"]).exists()
]

if missing_models:
    st.error(
        "Saved model files are missing: "
        + ", ".join(missing_models)
        + ". Run train_models.py first."
    )
    st.stop()


# --------------------------------------------------
# 5. Sidebar: model selection and sample download
# --------------------------------------------------
st.sidebar.header("Choose a model")

selected_model_key = st.sidebar.selectbox(
    "Classification model",
    options=list(model_details.keys()),
    format_func=lambda key: model_details[key]["display_name"],
)

st.sidebar.write(
    "All models were trained beforehand. "
    "Uploading a CSV does not retrain them."
)

if TEST_DATA_PATH.exists():
    st.sidebar.download_button(
        label="Download sample test CSV",
        data=TEST_DATA_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# 6. Upload test CSV
# --------------------------------------------------
st.subheader("1. Upload test data")

st.write(
    f"The CSV must include all {len(feature_names)} input features "
    f"and a '{target_column}' column containing 0 or 1."
)

with st.expander("Show required input columns"):
    st.write(feature_names)

uploaded_file = st.file_uploader(
    "Choose your test CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "Download the sample test CSV from the sidebar, "
        "then upload it here to begin."
    )
    st.stop()


# --------------------------------------------------
# 7. Validate uploaded data
# --------------------------------------------------
try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

required_columns = feature_names + [target_column]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        "Missing required columns: " + ", ".join(missing_columns)
    )
    st.stop()

try:
    # Select features in exactly the training-time order
    X = data[feature_names].apply(
        pd.to_numeric,
        errors="raise",
    )

    y = pd.to_numeric(
        data[target_column],
        errors="raise",
    )
except (ValueError, TypeError):
    st.error(
        "Input features and target must contain numeric values."
    )
    st.stop()

if not np.isfinite(X.to_numpy(dtype=float)).all():
    st.error(
        "Input features contain missing or infinite values."
    )
    st.stop()

if y.isna().any() or not y.isin([0, 1]).all():
    st.error(
        "The target column must contain only 0 (Benign) "
        "or 1 (Malignant), with no missing values."
    )
    st.stop()

y = y.astype(int)

st.success(
    f"Validated {len(X)} test rows and "
    f"{len(feature_names)} input features."
)

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10))

if y.nunique() < 2:
    st.warning(
        "This CSV contains only one target class. "
        "AUC cannot be calculated and will appear as N/A. "
        "Other metrics should be interpreted with this limitation."
    )


# --------------------------------------------------
# 8. Evaluate all five saved models
# --------------------------------------------------
results = []
model_outputs = {}

try:
    for model_key, info in model_details.items():
        model_path = MODEL_DIR / info["filename"]
        model = load_model(str(model_path))

        metrics, predictions, probabilities = calculate_metrics(
            model, X, y
        )

        results.append({
            "Model": info["display_name"],
            **metrics,
        })

        model_outputs[model_key] = {
            "metrics": metrics,
            "predictions": predictions,
            "probabilities": probabilities,
        }

except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()


# --------------------------------------------------
# 9. Show selected model's metrics
# --------------------------------------------------
selected_name = model_details[selected_model_key]["display_name"]
selected_output = model_outputs[selected_model_key]

st.subheader(f"2. Results: {selected_name}")

metric_columns = st.columns(3)

for index, (name, value) in enumerate(
    selected_output["metrics"].items()
):
    display_value = (
        "N/A" if pd.isna(value) else f"{value:.4f}"
    )

    metric_columns[index % 3].metric(
        label=name,
        value=display_value,
    )

st.caption(
    "Precision, recall and F1 use Malignant (1) as the positive class."
)


# --------------------------------------------------
# 10. Confusion matrix and classification report
# --------------------------------------------------
st.subheader("3. Prediction breakdown")

matrix_tab, report_tab = st.tabs([
    "Confusion matrix",
    "Classification report",
])

with matrix_tab:
    matrix = confusion_matrix(
        y,
        selected_output["predictions"],
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(6, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Benign", "Malignant"],
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title(selected_name)
    figure.tight_layout()

    st.pyplot(figure)
    plt.close(figure)

with report_tab:
    report = classification_report(
        y,
        selected_output["predictions"],
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4)
    )


# --------------------------------------------------
# 11. Comparison table for all models
# --------------------------------------------------
st.subheader("4. Compare all five models")

comparison = pd.DataFrame(results).set_index("Model")

st.dataframe(
    comparison.style.format(
        "{:.4f}",
        na_rep="N/A",
    )
)

st.download_button(
    label="Download comparison results",
    data=comparison.to_csv().encode("utf-8"),
    file_name="uploaded_test_comparison.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 12. Show and download individual predictions
# --------------------------------------------------
with st.expander("Show individual predictions"):
    prediction_table = pd.DataFrame({
        "Actual target": y.to_numpy(),
        "Predicted target": selected_output["predictions"],
        "Malignant probability": selected_output["probabilities"],
    })

    st.dataframe(prediction_table.round(4))

    st.download_button(
        label="Download selected model predictions",
        data=prediction_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model_key}_predictions.csv",
        mime="text/csv",
    )