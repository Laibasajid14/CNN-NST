"""
Evaluation functions for consistent metric calculation across assignments.
"""

import numpy as np
import pandas as pd
import json

def calculate_metrics(predicted_counts, actual_counts):
    """
    Calculate evaluation metrics
    Args:
        predicted_counts: List or array of predicted counts
        actual_counts: List or array of ground truth counts
    """
    predicted = np.array(predicted_counts)
    actual = np.array(actual_counts)
    
    # Handle zero actual counts to avoid division by zero
    mask = actual > 0
    
    errors = np.abs(predicted - actual)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    # Accuracy: percentage of images within 10% error
    pct_error = np.zeros_like(actual, dtype=float)
    pct_error[mask] = (errors[mask] / actual[mask]) * 100
    
    accurate = np.sum(pct_error <= 10)
    accuracy_pct = (accurate / len(actual)) * 100
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'accuracy_percentage': float(accuracy_pct),
        'mean_pct_error': float(np.mean(pct_error)),
        'std_pct_error': float(np.std(pct_error)),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors))
    }

def load_ground_truth(csv_path):
    """Load ground truth counts from CSV"""
    df = pd.read_csv(csv_path)
    return dict(zip(df['filename'], df['actual_count']))
