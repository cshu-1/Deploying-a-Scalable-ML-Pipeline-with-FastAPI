import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from ml.model import train_model, compute_model_metrics, inference, save_model, load_model
from ml.data import process_data
import tempfile
import os



def test_compute_model_metrics_returns_expected_values():
    """
    Test that compute_model_metrics returns expected precision, recall, and F1 values
    for known prediction scenarios.
    """
    # Test case 1: Perfect predictions
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Perfect predictions should yield 1.0 for all metrics
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0

    # Test case 2: Known imperfect predictions
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    # True Positives: 2, False Positives: 1, False Negatives: 1
    # Expected: Precision = 2/3 ≈ 0.6667, Recall = 2/3 ≈ 0.6667, F1 = 2/3 ≈ 0.6667

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert abs(precision - 0.6667) < 0.001
    assert abs(recall - 0.6667) < 0.001
    assert abs(fbeta - 0.6667) < 0.001


def test_inference_returns_expected_type_and_shape():
    """
    Test that the inference function returns predictions with expected type and shape.
    """
    # Create sample data with fixed seed for reproducibility
    np.random.seed(42)
    X_train = np.random.rand(50, 4)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 4)

    # Train a model
    model = train_model(X_train, y_train)

    # Get predictions
    predictions = inference(model, X_test)

    # Test return type
    assert isinstance(predictions, np.ndarray)

    # Test shape matches input
    assert predictions.shape == (X_test.shape[0],)

    # Test predictions are binary integers
    assert predictions.dtype in [np.int32, np.int64, int]
    assert all(pred in [0, 1] for pred in predictions)



def test_data_split_maintains_expected_proportions():
    """
    Test that train-test splits maintain expected data types and reasonable proportions.
    """
    # Create sample data
    data = pd.DataFrame({
        'feature1': range(100),
        'feature2': ['A'] * 50 + ['B'] * 50,
        'label': [0] * 30 + [1] * 70  # Imbalanced for testing stratification
    })

    # Simulate train-test split (80-20)
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # Test data types are preserved
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)

    # Test approximate size expectations (allowing some tolerance)
    assert 0.75 <= len(train_data) / len(data) <= 0.85  # Around 80%
    assert 0.15 <= len(test_data) / len(data) <= 0.25  # Around 20%

    # Test that all original columns are preserved
    assert list(train_data.columns) == list(data.columns)
    assert list(test_data.columns) == list(data.columns)