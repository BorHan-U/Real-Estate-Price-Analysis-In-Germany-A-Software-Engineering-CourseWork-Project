"""
Unit tests for model_evaluation module.
"""

import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.model_evaluation import model_evaluation


class TestModelEvaluation(unittest.TestCase):
    """
    Test case for the model_evaluation function.
    """

    def test_evaluation(self):
        """
        Test model evaluation for a LinearRegression model.
        """
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
        model = LinearRegression().fit(X, y)
        metrics = model_evaluation('LinearRegression', model, X, y, 'output.csv')
        self.assertIn('MSE', metrics)
        self.assertIn('R2-Score', metrics)


if __name__ == '__main__':
    unittest.main()
