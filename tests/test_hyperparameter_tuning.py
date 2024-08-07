"""
Unit tests for hyperparameter_tuning module.
"""

import unittest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.hyperparameter_tuning import hyperparameter_tuning


class TestHyperparameterTuning(unittest.TestCase):
    """
    Test case for the hyperparameter_tuning function.
    """

    def test_tuning(self):
        """
        Test hyperparameter tuning for a RandomForest model.
        """
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        models = [('RandomForest', RandomForestClassifier())]
        param_grids = [{'n_estimators': [10, 50], 'max_depth': [None, 10]}]
        best_models, best_params = hyperparameter_tuning(models, param_grids, X, y)
        self.assertIn('RandomForest', best_models)
        self.assertIn('RandomForest', best_params)


if __name__ == '__main__':
    unittest.main()
