"""
Unit tests for drop_columns_with_zero_threshold module.

This module contains comprehensive tests to ensure the correct functionality
of the drop_columns_with_zero_threshold function under various scenarios,
including edge cases and unexpected inputs.
"""

import unittest
import pandas as pd
import numpy as np
from modules.drop_columns_with_zero_threshold import (
    drop_columns_with_zero_threshold
)


class TestDropColumnsWithZeroThreshold(unittest.TestCase):
    """
    Test case for the drop_columns_with_zero_threshold function.

    This class contains various test methods
    to ensure the correct functionality
    of the drop_columns_with_zero_threshold
    function under different scenarios,
    including normal operations, edge cases, and error handling.
    """

    def test_drop_columns_normal(self):
        """
        Test dropping of columns with zero data above
        a threshold in a normal scenario.
        """
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [1, 2, 3],
            'C': [0, 1, 0]
        })
        threshold = 2
        result = drop_columns_with_zero_threshold(data, threshold)
        expected_data = pd.DataFrame({
            'B': [1, 2, 3],
            'C': [0, 1, 0]
        })
        pd.testing.assert_frame_equal(result, expected_data)

    def test_empty_dataframe(self):
        """
        Test handling of an empty DataFrame.
        """
        data = pd.DataFrame()
        threshold = 2
        with self.assertRaises(ValueError):
            drop_columns_with_zero_threshold(data, threshold)

    def test_negative_threshold(self):
        """
        Test handling of a negative threshold.
        """
        data = pd.DataFrame({'A': [1, 2, 3]})
        threshold = -1
        with self.assertRaises(ValueError):
            drop_columns_with_zero_threshold(data, threshold)

    def test_no_numeric_columns(self):
        """
        Test handling of a DataFrame with no numeric columns.
        """
        data = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': ['d', 'e', 'f']
        })
        threshold = 2
        result = drop_columns_with_zero_threshold(data, threshold)
        pd.testing.assert_frame_equal(result, data)

    def test_mixed_data_types(self):
        """
        Test handling of mixed data types.
        """
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': ['a', 'b', 'c'],
            'C': [1, 2, 3],
            'D': [True, False, True]
        })
        threshold = 2
        result = drop_columns_with_zero_threshold(data, threshold)
        expected_data = pd.DataFrame({
            'B': ['a', 'b', 'c'],
            'C': [1, 2, 3],
            'D': [True, False, True]
        })
        pd.testing.assert_frame_equal(result, expected_data)

    def test_nan_values(self):
        """
        Test handling of NaN values.
        """
        data = pd.DataFrame({
            'A': [0, np.nan, 0],
            'B': [1, 2, 3],
            'C': [np.nan, np.nan, np.nan]
        })
        threshold = 2
        result = drop_columns_with_zero_threshold(data, threshold)
        expected_data = pd.DataFrame({
            'A': [0, np.nan, 0],
            'B': [1, 2, 3],
            'C': [np.nan, np.nan, np.nan]
        })
        pd.testing.assert_frame_equal(result, expected_data)

    def test_all_columns_dropped(self):
        """
        Test the case where all columns are dropped.
        """
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [0, 0, 0],
            'C': [0, 0, 0]
        })
        threshold = 2
        result = drop_columns_with_zero_threshold(data, threshold)
        self.assertTrue(result.empty)

    def test_no_columns_dropped(self):
        """
        Test the case where no columns are dropped.
        """
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        threshold = 0
        result = drop_columns_with_zero_threshold(data, threshold)
        pd.testing.assert_frame_equal(result, data)

    def test_boolean_column(self):
        """
        Test handling of boolean columns.
        """
        data = pd.DataFrame({
            'A': [True, False, True],
            'B': [False, False, False],
            'C': [1, 2, 3]
        })
        threshold = 2
        result = drop_columns_with_zero_threshold(data, threshold)
        expected_data = pd.DataFrame({
            'A': [True, False, True],
            'C': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(result, expected_data)


if __name__ == '__main__':
    unittest.main()
