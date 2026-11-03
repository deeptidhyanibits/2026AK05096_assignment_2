Dataset = https://www.kaggle.com/datasets/utkarshx27/breast-cancer-wisconsin-diagnostic-dataset?resource=download

# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement

The objective of this project is to classify breast tumour samples
as benign or malignant using numerical measurements of cell nuclei.

Five classification models are trained on the same training dataset
and evaluated on the same held-out test dataset. Their performance
is compared using Accuracy, AUC, Precision, Recall, F1 Score, and
Matthews Correlation Coefficient (MCC).

An interactive Streamlit application allows users to upload labelled
test data, select a saved model, and view evaluation results without
retraining the models.

## b. Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)

Original source:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Local dataset filename: training_data.csv

| Property | Description |
|---|---|
| Number of instances | 569 |
| Number of input features | 30 |
| Classification task | Binary classification |
| Original target column | y |
| Original target labels | B = Benign, M = Malignant |
| Encoded target | 0 = Benign, 1 = Malignant |
| Benign instances | 357 |
| Malignant instances | 212 |
| Training instances | 455 |
| Test instances | 114 |

The features describe cell-nucleus characteristics such as radius,
texture, perimeter, area, smoothness, compactness, and concavity.
Measurements include mean, standard error, and worst values.

The unnamed row-number column is excluded from the input features.
The target column is separated from the features.

The dataset is divided into 80% training data and 20% test data using
a stratified split with random_state=42.

StandardScaler is included in the Logistic Regression, KNN, and SVM
pipelines. Scaling is fitted only on the training data. Decision Tree
and Gaussian Naive Bayes use the unscaled input features.

## c. GitHub Repository Link

[Open the project repository](PASTE_YOUR_GITHUB_REPOSITORY_URL_HERE)

## d. Models Used and Evaluation Results

All five models are evaluated on the same held-out test dataset.
Precision, Recall, and F1 use Malignant (1) as the positive class.
AUC is calculated using predicted probabilities for the malignant class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | TODO | TODO | TODO | TODO | TODO | TODO |
| Decision Tree | TODO | TODO | TODO | TODO | TODO | TODO |
| KNN | TODO | TODO | TODO | TODO | TODO | TODO |
| Naive Bayes | TODO | TODO | TODO | TODO | TODO | TODO |
| Support Vector Machine | TODO | TODO | TODO | TODO | TODO | TODO |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TODO |
| Decision Tree | TODO |
| KNN | TODO |
| Naive Bayes | TODO |
| Support Vector Machine | TODO |
| Overall winner for this dataset | TODO |

## f. How to Run the Project

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Train the models

Ensure training_data.csv is present in the project root, then run:

```bash
python train_models.py
```

This creates the saved model pipelines, feature schema, held-out
test_data.csv, and evaluation outputs.

### Run the Streamlit app

```bash
python -m streamlit run app.py
```

Open the address displayed in the terminal.

Upload the generated test_data.csv, which contains all 30 input
features and the target column with labels 0 and 1.

Select a model to view its metrics, confusion matrix, and
classification report. The app also displays a comparison of all
five saved models.

The app loads saved models and does not retrain on uploaded data.

## g. Live Streamlit Application

[Open the Streamlit app](PASTE_YOUR_STREAMLIT_APP_URL_HERE)