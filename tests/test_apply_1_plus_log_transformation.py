"""
This module contains unit tests for the apply_1_plus_log_transformation function.

The tests cover basic functionality, edge cases, error handling, and input validation
for the log(1+x) transformation function.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from modules.apply_1_plus_log_transformation import apply_1_plus_log_transformation


class TestApply1PlusLogTransformation(unittest.TestCase):
    """
    Test case for the apply_1_plus_log_transformation function.

    This class contains various test methods to ensure the correct functionality
    of the apply_1_plus_log_transformation function under different scenarios.
    """

    def test_transformation(self):
        """
        Test the log transformation on specified columns.

        This test verifies that the function correctly applies the log(1+x)
        transformation to the specified columns of a DataFrame.
        """
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        columns_to_transform = ['A', 'B']
        transformed_data = apply_1_plus_log_transformation(data, columns_to_transform)
        expected_data = pd.DataFrame({
            'A': np.log1p([1, 2, 3]),
            'B': np.log1p([4, 5, 6])
        })
        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_empty_dataframe(self):
        """
        Test the function with an empty DataFrame.

        This test checks if the function correctly handles an empty DataFrame input
        by returning an empty DataFrame.
        """
        data = pd.DataFrame()
        columns_to_transform = ['A', 'B']
        transformed_data = apply_1_plus_log_transformation(data, columns_to_transform)
        self.assertTrue(transformed_data.empty)

    def test_missing_columns(self):
        """
        Test the function when specified columns are not in the DataFrame.

        This test verifies that the function raises a ValueError when asked to
        transform a column that doesn't exist in the input DataFrame.
        """
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        columns_to_transform = ['A', 'C']
        with self.assertRaises(ValueError):
            apply_1_plus_log_transformation(data, columns_to_transform)

    def test_non_numeric_data(self):
        """
        Test the function when non-numeric data is present in columns to transform.

        This test checks if the function raises a ValueError when encountering
        non-numeric data in a column that is to be transformed.
        """
        data = pd.DataFrame({
            'A': [1, 2, 'three'],
            'B': [4, 5, 6]
        })
        columns_to_transform = ['A', 'B']
        with self.assertRaises(ValueError):
            apply_1_plus_log_transformation(data, columns_to_transform)

    def test_invalid_input_type(self):
        """
        Test the function when input is not a DataFrame.

        This test verifies that the function raises a TypeError when given
        input that is not a pandas DataFrame.
        """
        data = [1, 2, 3]
        columns_to_transform = ['A', 'B']
        with self.assertRaises(TypeError):
            apply_1_plus_log_transformation(data, columns_to_transform)


if __name__ == '__main__':
    unittest.main()
