"""
Unit tests for hyperparameter_tuning module.

This module contains comprehensive tests to ensure the correct functionality
of the hyperparameter_tuning function under various scenarios, including
edge cases and unexpected inputs.
"""
import unittest
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from modules.hyperparameter_tuning import hyperparameter_tuning


class TestHyperparameterTuning(unittest.TestCase):
    """
    Test case for the hyperparameter_tuning function.

    This class contains various test methods
    to ensure the correct functionality
    of the hyperparameter_tuning function under
    different scenarios, including
    normal operations, edge cases, and error handling.
    """

    def setUp(self):
        """
        Set up test data and models.
        """
        self.x, self.y = make_regression(
            n_samples=100, n_features=20, noise=0.1, random_state=42)[:2]
        self.x = pd.DataFrame(self.x)
        self.y = pd.Series(self.y)

    def test_tuning_single_model(self):
        """
        Test hyperparameter tuning for a single RandomForest model.
        """
        models = [('RandomForest', RandomForestRegressor(random_state=42))]
        param_grids = [{'n_estimators': [10, 50], 'max_depth': [None, 10]}]
        best_models, best_params = hyperparameter_tuning(models,
                                                         param_grids,
                                                         self.x,
                                                         self.y)
        self.assertIn('RandomForest', best_models)
        self.assertIn('RandomForest', best_params)
        self.assertIsInstance(best_models['RandomForest'],
                              RandomForestRegressor)
        self.assertIsInstance(best_params['RandomForest'], dict)

    def test_tuning_multiple_models(self):
        """
        Test hyperparameter tuning for multiple models.
        """
        models = [
            ('RandomForest', RandomForestRegressor(random_state=42)),
            ('LinearRegression', LinearRegression())
        ]
        param_grids = [
            {'n_estimators': [10, 50], 'max_depth': [None, 10]},
            {'fit_intercept': [True, False]}
        ]
        best_models, best_params = hyperparameter_tuning(
            models, param_grids, self.x, self.y)
        self.assertEqual(len(best_models), 2)
        self.assertEqual(len(best_params), 2)
        self.assertIn('RandomForest', best_models)
        self.assertIn('LinearRegression', best_models)

    def test_empty_models_and_param_grids(self):
        """
        Test handling of empty models and param_grids lists.
        """
        with self.assertRaises(ValueError):
            hyperparameter_tuning([], [], self.x, self.y)

    def test_mismatched_models_and_param_grids(self):
        """
        Test handling of mismatched lengths of models and param_grids lists.
        """
        models = [('RandomForest', RandomForestRegressor())]
        param_grids = [{'n_estimators': [10, 50]}, {'max_depth': [None, 10]}]
        with self.assertRaises(ValueError):
            hyperparameter_tuning(models, param_grids, self.x, self.y)

    def test_invalid_model(self):
        """
        Test handling of an invalid model that doesn't
        implement the scikit-learn estimator interface.
        """
        class InvalidModel:
            """
            A mock model class for testing purposes.
            """
            def fit(self, x, y):
                """
                A mock fit method
                """

            def predict(self, x):
                """
                A mock predict method.
                """
                return [0] * len(x)

        models = [('InvalidModel', InvalidModel())]
        param_grids = [{}]
        best_models, best_params = hyperparameter_tuning(models,
                                                         param_grids,
                                                         self.x, self.y)
        self.assertIsNone(best_models['InvalidModel'])
        self.assertIsNone(best_params['InvalidModel'])

    def test_empty_param_grid(self):
        """
        Test handling of an empty parameter grid.
        """
        models = [('LinearRegression', LinearRegression())]
        param_grids = [{}]
        best_models, best_params = hyperparameter_tuning(models,
                                                         param_grids,
                                                         self.x,
                                                         self.y)
        self.assertIn('LinearRegression', best_models)
        self.assertIn('LinearRegression', best_params)
        self.assertEqual(best_params['LinearRegression'], {})

    def test_non_dataframe_input(self):
        """
        Test handling of non-DataFrame input.
        """
        x = self.x.values
        y = self.y.values
        models = [('RandomForest', RandomForestRegressor(random_state=42))]
        param_grids = [{'n_estimators': [10, 50]}]
        best_models, best_params = hyperparameter_tuning(models,
                                                         param_grids,
                                                         x, y)
        self.assertIn('RandomForest', best_models)
        self.assertIn('RandomForest', best_params)


if __name__ == '__main__':
    unittest.main()
