"""Metrics and evaluation utilities."""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred), #(y_true, y_pred, average='weighted', zero_division=0)
        'recall': recall_score(y_true, y_pred), #(y_true, y_pred, average='weighted', zero_division=0)
        'f1': f1_score(y_true, y_pred) #(y_true, y_pred, average='weighted', zero_division=0)
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print detailed classification report."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)
    
    metrics = calculate_metrics(y_true, y_pred)
    for name, value in metrics.items():
        print(f"{name.capitalize():12s}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("DETAILED REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred))