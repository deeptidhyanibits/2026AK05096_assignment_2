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
https://www.kaggle.com/datasets/utkarshx27/breast-cancer-wisconsin-diagnostic-dataset?resource=download

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

[Open the project repository](https://github.com/deeptidhyanibits/2026AK05096_assignment_2)

## d. Models Used and Evaluation Results

All five models were trained on the same training dataset and
evaluated on the same held-out test dataset of 114 instances.

Precision, Recall, and F1 use Malignant (1) as the positive class.
AUC is calculated using predicted probabilities for the malignant
class. All scores are reported as decimals.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## e. Observations on Model Performance

The following observations are based on the held-out test set of
114 instances. Malignant (1) is the positive class.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 0.9649 and the highest AUC of 0.9960. Its recall of 0.9286 matched SVM, with 3 malignant cases missed and 1 benign case incorrectly classified as malignant. |
| Decision Tree | Achieved an accuracy of 0.9211 and an F1 score of 0.8861, the lowest among the five models. It missed 7 malignant cases and incorrectly classified 2 benign cases as malignant. |
| KNN | Achieved an accuracy of 0.9561, precision of 0.9744, and F1 score of 0.9383. It missed 4 malignant cases and incorrectly classified 1 benign case as malignant. |
| Naive Bayes | Achieved perfect precision of 1.0000, with no benign cases incorrectly classified as malignant. However, its recall of 0.8333 means it missed 7 malignant cases, resulting in an F1 score of 0.9091. |
| Support Vector Machine | Achieved the highest accuracy of 0.9737, F1 score of 0.9630, and MCC of 0.9442. It correctly classified all benign cases and detected 39 of 42 malignant cases, missing 3. |
| Overall winner for this dataset | SVM was selected as the overall winner on this test set because it achieved the highest accuracy, F1, and MCC, while matching the highest recall and achieving perfect precision. Logistic Regression had a slightly higher AUC, indicating stronger ranking performance across classification thresholds. |

These findings apply to this particular train/test split and do not
establish that one model will perform best on every unseen dataset.

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

[Open the Streamlit app](https://2026ak05096assignment2.streamlit.app)