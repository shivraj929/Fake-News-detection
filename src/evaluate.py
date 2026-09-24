"""
Model Evaluation Module
- Loads trained models and test data
- Computes accuracy, precision, recall, F1-score
- Generates confusion matrices
- Produces a comparison summary
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import scipy.sparse
import joblib
import os

import config


def load_test_data():
    """Load saved test data."""
    X_test = scipy.sparse.load_npz("models/X_test.npz")
    y_test = pd.read_csv("models/y_test.csv").squeeze()
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
    }


def evaluate_all_models():
    """Evaluate all 4 trained models and produce a comparison table."""
    X_test, y_test = load_test_data()
    print(f"Test set size: {len(y_test)}")

    models_info = {
        "Logistic Regression": config.LOGISTIC_MODEL,
        "Naive Bayes": config.NAIVE_BAYES_MODEL,
        "SVM (LinearSVC)": config.SVM_MODEL,
        "Random Forest": config.RANDOM_FOREST_MODEL,
    }

    results = []

    for name, path in models_info.items():
        if not os.path.exists(path):
            print(f"\nWARNING: {path} not found. Skipping {name}.")
            continue

        model = joblib.load(path)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    # Summary comparison table
    if results:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("F1-Score", ascending=False)
        print(results_df.to_string(index=False))

        # Save results
        results_df.to_csv("reports/model_comparison.csv", index=False)
        print("\nResults saved to reports/model_comparison.csv")

        # Best model
        best = results_df.iloc[0]
        print(f"\n** Best Model: {best['Model']} (F1-Score: {best['F1-Score']:.4f})")

    return results


if __name__ == "__main__":
    evaluate_all_models()
