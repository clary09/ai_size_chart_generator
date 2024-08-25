# tests/test_accuracy_validation.py
import pytest
from src.accuracy_validation import AccuracyValidator
import pandas as pd

@pytest.fixture
def known_size_chart():
    return pd.DataFrame({
        'size': ['S', 'M', 'L', 'XL'],
        'height': [160, 170, 180, 190],
        'weight': [50, 70, 90, 110],
        'chest': [80, 90, 100, 110],
        'waist': [60, 70, 80, 90],
        'hip': [90, 100, 110, 120]
    })

@pytest.fixture
def generated_size_chart():
    return pd.DataFrame({
        'size': ['S', 'M', 'L', 'XL'],
        'height': [161, 169, 181, 189],
        'weight': [51, 69, 89, 111],
        'chest': [81, 89, 99, 111],
        'waist': [61, 69, 79, 91],
        'hip': [91, 99, 109, 121]
    })

def test_accuracy_validation(known_size_chart, generated_size_chart):
    validator = AccuracyValidator(known_size_chart)
    mse = validator.validate(generated_size_chart)
    assert mse < 2, f"MSE too high: {mse}"
