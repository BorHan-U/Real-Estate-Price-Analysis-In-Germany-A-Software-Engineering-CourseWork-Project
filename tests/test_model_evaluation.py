"""
Unit tests for model_evaluation module.

This module contains comprehensive tests to ensure the correct functionality
of the model_evaluation function under various scenarios, including
edge cases and error handling.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import sys
import os
import tempfile

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.model_evaluation import model_evaluation, ModelEvaluationError


class TestModelEvaluation(unittest.TestCase):
    """
    Test case for the model_evaluation function.

    This class contains various test methods to ensure the correct functionality
    of the model_evaluation function under different scenarios, including
    normal operations, edge cases, and error handling.
    """

    def setUp(self):
        """Set up test data and model."""
        self.X, self.y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
        self.model = LinearRegression().fit(self.X, self.y)

    def test_evaluation_normal(self):
        """Test model evaluation for a LinearRegression model with normal input."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            metrics = model_evaluation('LinearRegression', self.model, self.X, self.y, temp_file.name)
        
        self.assertIn('MSE', metrics)
        self.assertIn('R2-Score', metrics)
        self.assertGreater(metrics['R2-Score'], 0)  # R2 should be positive for this regression
        
        # Check if the output file was created and contains the correct data
        with open(temp_file.name, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[0].strip(), "Predicted,Actual")
        self.assertEqual(len(lines), 101)  # Header + 100 data points
        
        os.unlink(temp_file.name)  # Clean up the temporary file

    def test_empty_input(self):
        """Test handling of empty input data."""
        with self.assertRaises(ValueError):
            model_evaluation('LinearRegression', self.model, np.array([]), np.array([]), 'output.csv')

    def test_mismatched_samples(self):
        """Test handling of mismatched number of samples in X and y."""
        with self.assertRaises(ValueError):
            model_evaluation('LinearRegression', self.model, self.X, self.y[:-1], 'output.csv')

    def test_invalid_model(self):
        """Test handling of an invalid model."""
        invalid_model = "Not a model"
        with self.assertRaises(ModelEvaluationError):
            model_evaluation('InvalidModel', invalid_model, self.X, self.y, 'output.csv')

    def test_non_writable_output(self):
        """Test handling of a non-writable output file."""
        with self.assertRaises(ModelEvaluationError):
            model_evaluation('LinearRegression', self.model, self.X, self.y, '/nonexistent/path/output.csv')

    def test_pandas_input(self):
        """Test handling of pandas DataFrame and Series as input."""
        X_df = pd.DataFrame(self.X)
        y_series = pd.Series(self.y)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            metrics = model_evaluation('LinearRegression', self.model, X_df, y_series, temp_file.name)
        
        self.assertIn('MSE', metrics)
        self.assertIn('R2-Score', metrics)
        
        os.unlink(temp_file.name)  # Clean up the temporary file

    def test_different_model(self):
        """Test evaluation with a different model (e.g., a dummy model that always predicts the mean)."""
        class DummyModel:
            def predict(self, X):
                return np.full(X.shape[0], np.mean(self.y))
            
            def fit(self, X, y):
                self.y = y
                return self

        dummy_model = DummyModel().fit(self.X, self.y)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            metrics = model_evaluation('DummyModel', dummy_model, self.X, self.y, temp_file.name)
        
        self.assertIn('MSE', metrics)
        self.assertIn('R2-Score', metrics)
        self.assertAlmostEqual(metrics['R2-Score'], 0, places=5)  # R2 should be close to 0 for this dummy model
        
        os.unlink(temp_file.name)  # Clean up the temporary file


if __name__ == '__main__':
    unittest.main()
