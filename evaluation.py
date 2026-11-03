import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted model. Target 1 represents malignant."""

    # Predicted class: 0 or 1
    predictions = model.predict(X_test)

    # Locate the probability column corresponding to malignant (class 1)
    positive_index = np.flatnonzero(model.classes_ == 1).item()
    probabilities = model.predict_proba(X_test)[:, positive_index]

    # AUC requires both target classes
    auc = (
        roc_auc_score(y_test, probabilities)
        if len(np.unique(y_test)) == 2
        else np.nan
    )

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
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